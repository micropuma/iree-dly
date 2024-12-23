// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// Utility methods
//===---------------------------------------------------------------------===//

// Utility to apply a tile-swizzling to a packed shape.
static SmallVector<OpFoldResult>
getSwizzledShape(ArrayRef<OpFoldResult> packedShape,
                 MaterializeEncodingInfo encodingInfo) {
  if (packedShape.empty() || !encodingInfo.swizzle) {
    return SmallVector<OpFoldResult>(packedShape);
  }

  int64_t srcRank = packedShape.size() - encodingInfo.innerTileSizes.size();
  SmallVector<int64_t> perm = llvm::to_vector(llvm::seq<int64_t>(0, srcRank));
  for (auto i : encodingInfo.swizzle->permutation) {
    perm.push_back(i + srcRank);
  }

  SmallVector<OpFoldResult> newShape(packedShape.take_front(srcRank));
  SmallVector<int64_t> expandedTileShape =
      getExpandedTileShape(encodingInfo.swizzle->expandShape);
  MLIRContext *ctx = packedShape[0].getContext();
  Builder b(ctx);
  for (int64_t d : expandedTileShape) {
    newShape.push_back(b.getIndexAttr(d));
  }
  applyPermutationToVector(newShape, perm);

  return newShape;
}

static Operation *dropEncodingAndCloneOp(OpBuilder &builder, Operation *op,
                                         ValueRange convertedInputOperands,
                                         ValueRange convertedOutputOperands) {
  SmallVector<Value> operands;
  operands.append(convertedInputOperands.begin(), convertedInputOperands.end());
  operands.append(convertedOutputOperands.begin(),
                  convertedOutputOperands.end());
  return mlir::clone(builder, op,
                     {dropEncoding(cast<RankedTensorType>(
                         convertedOutputOperands[0].getType()))},
                     operands);
}

static FailureOr<SmallVector<OpFoldResult>>
getInnerTileSizesOfr(OpBuilder &rewriter, Location loc,
                     RankedTensorType tensorType,
                     const MaterializeEncodingInfo &materializeEncodingInfo,
                     MaterializeEncodingValueFn materializeEncodingValueFn) {
  ArrayRef<int64_t> staticTileSizes = materializeEncodingInfo.innerTileSizes;
  if (llvm::all_of(staticTileSizes,
                   [](int64_t i) { return !ShapedType::isDynamic(i); })) {
    return getAsOpFoldResult(rewriter.getI64ArrayAttr(staticTileSizes));
  }
  assert(materializeEncodingValueFn &&
         "When dynamic tile sizes are generated, a MaterializeEncodingValueFn "
         "should be provided.");

  FailureOr<MaterializeEncodingValueInfo> materializeEncodingValueInfo =
      materializeEncodingValueFn(tensorType, rewriter, loc);
  if (failed(materializeEncodingValueInfo)) {
    return failure();
  }
  ArrayRef<Value> innerTileSizeValues =
      materializeEncodingValueInfo->innerTileSizes;

  SmallVector<OpFoldResult> result(staticTileSizes.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (ShapedType::isDynamic(staticTileSizes[i])) {
      result[i] = innerTileSizeValues[i];
    } else if (tensorType.isDynamicDim(i)) {
      result[i] =
          rewriter.create<arith::ConstantIndexOp>(loc, staticTileSizes[i])
              .getResult();
    } else {
      result[i] = rewriter.getI64IntegerAttr(staticTileSizes[i]);
    }
  }
  return result;
}

RankedTensorType getExpandedType(RankedTensorType type, bool isBatched,
                                 bool isTransposed,
                                 SmallVectorImpl<ReassociationIndices> &ri) {
  if (!isBatched) {
    ri.assign({{0, 1}, {2, 3}});
    if (!isTransposed) {
      return RankedTensorType::get(
          {1, type.getDimSize(0), 1, type.getDimSize(1)},
          type.getElementType());
    }
    return RankedTensorType::get({type.getDimSize(0), 1, type.getDimSize(1), 1},
                                 type.getElementType());
  }

  ri.assign({{0}, {1, 2}, {3, 4}});
  if (!isTransposed) {
    return RankedTensorType::get(
        {type.getDimSize(0), 1, type.getDimSize(1), 1, type.getDimSize(2)},
        type.getElementType());
  }
  return RankedTensorType::get(
      {type.getDimSize(0), type.getDimSize(1), 1, type.getDimSize(2), 1},
      type.getElementType());
}

/// Given an input Value and a desired output element type, create and return
/// an element-wise linalg::GenericOp that extends the input Value to the
/// output element type.
static Value createElementWiseExtUIOp(RewriterBase &rewriter, Value input,
                                      Location loc, Type outElemType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto castedType = inputType.clone(outElemType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(rewriter, loc, input);
  Value init =
      rewriter.create<tensor::EmptyOp>(loc, inputMixedSizes, outElemType);
  return rewriter
      .create<linalg::GenericOp>(
          loc, castedType, input, init, maps, iteratorTypes,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value castRes =
                b.create<arith::ExtUIOp>(nestedLoc, outElemType, args[0])
                    ->getResult(0);
            b.create<linalg::YieldOp>(nestedLoc, castRes);
          })
      .getResult(0);
}

/// If needed, expand and the input Value, and return the resulting input with
/// the canonical mmt4d input shape. If the input element type is unsigned,
/// create a producer Linalg::GenericOp on the input that unsigned extends the
/// input to the output element type. This extension is required to keep the
/// unsignedness information on the input for ukernels. If `transpose` is true,
/// the `linalgOp`'s indexing maps are transposed.
static Value getMmt4dOperand(Value value, linalg::LinalgOp linalgOp,
                             bool transpose, RewriterBase &rewriter,
                             SmallVectorImpl<ReassociationIndices> &ri,
                             ArrayRef<Type> elemTypes, int operandIdx) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto cDims = linalg::inferContractionDims(linalgOp);
  Location loc = linalgOp->getLoc();
  Value expandedValue = value;
  // If vecmat with non-rhs operandIdx or matvec with non-lhs operandIdx, the
  // operand is a vector and must be extended
  if ((cDims->m.empty() && operandIdx != 1) ||
      (cDims->n.empty() && operandIdx != 0)) {
    auto type = cast<RankedTensorType>(value.getType());
    RankedTensorType newType = getExpandedType(
        type, /*isBatched=*/!cDims->batch.empty(),
        /*isTransposed=*/operandIdx == 2 && (transpose ^ cDims->n.empty()), ri);
    expandedValue =
        rewriter.create<tensor::ExpandShapeOp>(loc, newType, value, ri);
  }
  if (elemTypes[operandIdx].isUnsignedInteger()) {
    return createElementWiseExtUIOp(rewriter, expandedValue, loc,
                                    elemTypes.back());
  }
  return expandedValue;
}

static void transposeInPlace(MaterializeEncodingInfo &info) {
  // Vector cases: nothing to do.
  if (info.innerTileSizes.size() < 2) {
    return;
  }
  // Not a vector case, so all three arrays in `info` have size at least 2,
  // outerDimsPerm may have size 3 if there is a batch dimension, but in all
  // cases, the last 2 entries of each array are M and N, not batch.
  auto transpose = [](SmallVector<int64_t> &a) {
    std::swap(a[a.size() - 2], a[a.size() - 1]);
  };
  transpose(info.innerDimsPos);
  transpose(info.innerTileSizes);
  transpose(info.outerDimsPerm);
}

//===---------------------------------------------------------------------===//
// Methods to convert `set_encoding` and `unset_encoding` operations
// to `pack` and `unpack` operations respectively.
//===---------------------------------------------------------------------===//

/// TODO(hanchung): Move the implementation to EncodingUtils.cpp. It is not
/// moved because it needs some cleanup for this file. E.g., `getPaddingValue`
/// is no longer needed. Ideally we should move CPU specific patterns (e.g.,
/// lowerContractionOpWithEncoding, etc) to the CPUMaterializeEncoding file;
/// move general patterns to EncodingUtils, and retire this file.
FailureOr<tensor::PackOp> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, IREE::Encoding::SetEncodingOp encodingOp,
    Value source, const MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  RankedTensorType resultType = encodingOp.getResultType();
  FailureOr<MaterializeEncodingInfo> encodingInfo =
      typeConverter.getEncodingInfo(resultType);
  if (failed(encodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled result encoding");
  }

  auto encoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!encoding) {
    return failure();
  }
  if (typeConverter.getTransposeNarrowN() && isNarrowNResult(encoding)) {
    transposeInPlace(*encodingInfo);
  }

  // Create `tensor.empty` operation for the result of the pack operation.
  Location loc = encodingOp.getLoc();
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr = getInnerTileSizesOfr(
      rewriter, loc, resultType, *encodingInfo, materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultType.getElementType()));
  SmallVector<OpFoldResult> sourceDims =
      tensor::getMixedSizes(rewriter, loc, source);
  SmallVector<OpFoldResult> resultDims = tensor::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr, encodingInfo->innerDimsPos,
      encodingInfo->outerDimsPerm);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  resultType.getElementType());
  return rewriter.create<tensor::PackOp>(
      loc, source, emptyOp, encodingInfo->innerDimsPos, *innerTileSizesOfr,
      paddingValue, encodingInfo->outerDimsPerm);
}

/// TODO(hanchung): Move the implementation to EncodingUtils.cpp. See the reason
/// in the implementation comment of lowerSetEncodingToPackOp method.
FailureOr<tensor::UnPackOp> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, IREE::Encoding::UnsetEncodingOp encodingOp,
    Value packedValue, const MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  RankedTensorType sourceType = encodingOp.getSourceType();
  FailureOr<MaterializeEncodingInfo> encodingInfo =
      typeConverter.getEncodingInfo(sourceType);
  if (failed(encodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled source encoding");
  }
  auto encoding = IREE::Encoding::getEncodingAttr(sourceType);
  if (typeConverter.getTransposeNarrowN() && isNarrowNResult(encoding)) {
    transposeInPlace(*encodingInfo);
  }
  // Create an `tensor.empty` for the result of the unpack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> resultDims =
      getMixedValues(encodingOp.getResultType().getShape(),
                     encodingOp.getResultDims(), rewriter);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  sourceType.getElementType());
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr = getInnerTileSizesOfr(
      rewriter, loc, sourceType, *encodingInfo, materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  return rewriter.create<tensor::UnPackOp>(
      loc, packedValue, emptyOp, encodingInfo->innerDimsPos, *innerTileSizesOfr,
      encodingInfo->outerDimsPerm);
}

static FailureOr<Operation *> lowerContractionOpWithEncoding(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp, ValueRange operands,
    const MaterializeEncodingTypeConverter &typeConverter) {
  if (!linalgOp.hasPureTensorSemantics())
    return failure();

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return failure();
  }

  FailureOr<MaterializeEncodingInfo> encodingInfo =
      typeConverter.getEncodingInfo(
          cast<RankedTensorType>(linalgOp->getResultTypes()[0]));

  Operation *result;
  if (failed(encodingInfo)) {
    result = dropEncodingAndCloneOp(rewriter, linalgOp,
                                    operands.take_front(inputs.size()),
                                    operands.drop_front(inputs.size()));
  } else {
    bool transpose =
        typeConverter.getTransposeNarrowN() && isNarrowNResult(resultEncoding);
    SmallVector<Type> elemTypes = lhsEncoding.getElementTypesArray();
    SmallVector<ReassociationIndices> ri;
    Value newLhs = getMmt4dOperand(operands[0], linalgOp, transpose, rewriter,
                                   ri, elemTypes, /*operandIdx=*/0);
    Value newRhs = getMmt4dOperand(operands[1], linalgOp, transpose, rewriter,
                                   ri, elemTypes, /*operandIdx=*/1);
    Value newResult =
        getMmt4dOperand(operands[2], linalgOp, transpose, rewriter, ri,
                        elemTypes, /*operandIdx=*/2);
    if (transpose) {
      std::swap(newLhs, newRhs);
    }
    Type newResultType = newResult.getType();
    auto cDims = IREE::Encoding::getEncodingContractionDims(lhsEncoding);
    if (cDims->batch.empty()) {
      result = rewriter.create<linalg::Mmt4DOp>(
          linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
          ValueRange{newResult});
    } else {
      result = rewriter.create<linalg::BatchMmt4DOp>(
          linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
          ValueRange{newResult});
    }
    if (!ri.empty()) {
      result = rewriter.create<tensor::CollapseShapeOp>(
          linalgOp->getLoc(), operands[2].getType(), result->getResult(0), ri);
    }
  }
  return result;
}

/// Utility method to convert `tensor.empty` with encoding to a `tensor.empty`
/// of the materialized type.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, tensor::EmptyOp emptyOp,
                    ValueRange convertedOperands,
                    const MaterializeEncodingTypeConverter &typeConverter,
                    MaterializeEncodingValueFn materializeEncodingValueFn) {
  auto emptyType = cast<RankedTensorType>(emptyOp->getResultTypes()[0]);
  FailureOr<MaterializeEncodingInfo> encodingInfo =
      typeConverter.getEncodingInfo(emptyType);
  Location loc = emptyOp.getLoc();
  if (failed(encodingInfo)) {
    Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, emptyOp.getMixedSizes(), emptyType.getElementType());
    return newEmptyOp;
  }

  if (typeConverter.getTransposeNarrowN() &&
      isNarrowNResult(IREE::Encoding::getEncodingAttr(emptyType))) {
    transposeInPlace(*encodingInfo);
  }

  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr = getInnerTileSizesOfr(
      rewriter, loc, emptyType, *encodingInfo, materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        emptyOp, "failed to generate runtime tile size query");
  }

  SmallVector<OpFoldResult> sourceDims = emptyOp.getMixedSizes();
  (void)foldDynamicIndexList(sourceDims);
  SmallVector<OpFoldResult> newShape = tensor::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr, encodingInfo->innerDimsPos,
      encodingInfo->outerDimsPerm);
  newShape = getSwizzledShape(newShape, *encodingInfo);
  Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
      loc, newShape, emptyType.getElementType());
  return newEmptyOp;
}

/// Converts a linalg::GenericOp with encoded inputs into the packed domain.
/// The `genericOp` must have all parallel iterator types and a single output
/// with an identity indexing map.
static FailureOr<Operation *> lowerGenericOpWithEncoding(
    RewriterBase &rewriter, linalg::GenericOp genericOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    const MaterializeEncodingTypeConverter &typeConverter) {
  OpOperand *outputOperand = genericOp.getDpsInitOperand(0);
  AffineMap outputMap = genericOp.getMatchingIndexingMap(outputOperand);
  if (!outputMap.isIdentity()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Output indexing map is not identity");
  }
  FailureOr<MaterializeEncodingInfo> outMaterializeEncodingInfo =
      typeConverter.getEncodingInfo(
          cast<RankedTensorType>(outputOperand->get().getType()));
  if (failed(outMaterializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(
        genericOp, "MaterializeEncodingInfo failed for output");
  }

  auto convertedResultType =
      cast<RankedTensorType>(convertedOutputOperands[0].getType());
  SmallVector<utils::IteratorType> iteratorTypes(convertedResultType.getRank(),
                                                 utils::IteratorType::parallel);
  // Compute the new indexing maps for the packed layout. This assumes that
  // the output map is identity, and that all iterator types are parallel.
  SmallVector<int64_t> outInnerDimsPos =
      outMaterializeEncodingInfo->innerDimsPos;
  SmallVector<int64_t> outInverseOuterDimsPerm =
      invertPermutationVector(outMaterializeEncodingInfo->outerDimsPerm);
  SmallVector<AffineMap> packedIndexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
        typeConverter.getEncodingInfo(
            cast<RankedTensorType>(inputOperand->get().getType()));
    if (failed(materializeEncodingInfo)) {
      return rewriter.notifyMatchFailure(
          genericOp, "MaterializeEncodingInfo failed for input");
    }
    SmallVector<int64_t> innerDimsPos = materializeEncodingInfo->innerDimsPos;
    SmallVector<int64_t> outerDimsPerm = materializeEncodingInfo->outerDimsPerm;
    AffineMap inputMap = genericOp.getMatchingIndexingMap(inputOperand);
    // Permute result dims to the input packed domain, and map dims to the
    // output packed domain.
    SmallVector<int64_t> packedResultDims = llvm::map_to_vector(
        applyPermutation(inputMap.getResults(), outerDimsPerm),
        [&](AffineExpr expr) {
          auto dimExpr = cast<AffineDimExpr>(expr);
          return outInverseOuterDimsPerm[dimExpr.getPosition()];
        });
    // Add new dims for the inner tiles, taking the dim position from the
    // corresponding inner tile of the init operand.
    for (auto [idx, pos] : llvm::enumerate(innerDimsPos)) {
      auto dimPos = cast<AffineDimExpr>(inputMap.getResult(pos)).getPosition();
      for (auto [tileIdx, outDim] : llvm::enumerate(outInnerDimsPos)) {
        if (dimPos == outDim) {
          packedResultDims.push_back(outputMap.getNumDims() + tileIdx);
        }
      }
    }
    // Create the packed indexing map.
    SmallVector<AffineExpr> packedResultExprs =
        llvm::map_to_vector(packedResultDims, [&](int64_t dim) {
          return rewriter.getAffineDimExpr(dim);
        });
    auto packedInputMap = AffineMap::get(
        /*dimCount=*/iteratorTypes.size(), /*symbolCount=*/0, packedResultExprs,
        rewriter.getContext());
    packedIndexingMaps.push_back(packedInputMap);
  }
  // Create the new packed identity map for the output.
  packedIndexingMaps.push_back(
      rewriter.getMultiDimIdentityMap(convertedResultType.getRank()));
  auto materializedGenericOp = rewriter.create<linalg::GenericOp>(
      genericOp.getLoc(), convertedResultType, convertedInputOperands,
      convertedOutputOperands, packedIndexingMaps, iteratorTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(),
                              materializedGenericOp.getRegion(),
                              materializedGenericOp.getRegion().begin());
  return materializedGenericOp.getOperation();
}

/// Utility method to convert from a linalg::LinalgOp on `tensor` types with
/// encodings to a linalg::LinalgOp on the materialized type. The current
/// supported op types are:
///  - linalg::FillOp
///  - linalg::GenericOp
//   - All the iterators are parallel iterators.
//   - The op has a single output.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands,
                    const MaterializeEncodingTypeConverter &typeConverter,
                    MaterializeEncodingValueFn) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return rewriter.notifyMatchFailure(linalgOp, "Not pure tensor semantics");
  }
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops()) {
    return rewriter.notifyMatchFailure(linalgOp, "Loops are not all parallel");
  }
  if (linalgOp.getNumDpsInits() != 1) {
    return rewriter.notifyMatchFailure(linalgOp, "Not only 1 init operand");
  }

  return TypeSwitch<Operation *, FailureOr<Operation *>>(linalgOp)
      .Case<linalg::FillOp>(
          [&](linalg::FillOp fillOp) -> FailureOr<Operation *> {
            Operation *materializedFillOp = rewriter.create<linalg::FillOp>(
                fillOp.getLoc(), convertedOutputOperands[0].getType(),
                convertedInputOperands, convertedOutputOperands);
            return materializedFillOp;
          })
      .Case<linalg::GenericOp>(
          [&](linalg::GenericOp genericOp) -> FailureOr<Operation *> {
            return lowerGenericOpWithEncoding(
                rewriter, genericOp, convertedInputOperands,
                convertedOutputOperands, typeConverter);
          })
      .Default([](Operation *op) { return failure(); });
}

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the materialized shape of the `dispatchTensorType`. The
/// dynamic dimensions of the `dispatchTensorType` are provided in
/// `dynamicDims`.
static FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    const MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  auto boundTensorType =
      llvm::dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
  if (!boundTensorType) {
    return failure();
  }

  FailureOr<MaterializeEncodingInfo> encodingInfo =
      typeConverter.getEncodingInfo(boundTensorType);
  if (failed(encodingInfo)) {
    return failure();
  }
  if (typeConverter.getTransposeNarrowN() &&
      isNarrowNResult(IREE::Encoding::getEncodingAttr(boundTensorType))) {
    transposeInPlace(*encodingInfo);
  }

  SmallVector<OpFoldResult> targetShape =
      getMixedValues(boundTensorType.getShape(), dynamicDims, builder);
  auto innerTileSizes = getInnerTileSizesOfr(
      builder, loc, boundTensorType, *encodingInfo, materializeEncodingValueFn);
  if (failed(innerTileSizes)) {
    return failure();
  }
  SmallVector<OpFoldResult> convertedTargetShape =
      tensor::PackOp::getResultShape(builder, loc, targetShape, *innerTileSizes,
                                     encodingInfo->innerDimsPos,
                                     encodingInfo->outerDimsPerm);
  return getSwizzledShape(convertedTargetShape, *encodingInfo);
}

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the dynamic dimensions of the materialized shape of the
/// `dispatchTensorType`. The dynamic dimensions of the `dispatchTensorType` are
/// provided in `dynamicDims`.
static FailureOr<SmallVector<Value>> getPackedDynamicDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    const MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  FailureOr<SmallVector<OpFoldResult>> convertedTargetShape =
      getPackedDimsForDispatchTensor(builder, loc, typeConverter,
                                     dispatchTensorType, dynamicDims,
                                     materializeEncodingValueFn);
  if (failed(convertedTargetShape)) {
    return failure();
  }
  SmallVector<int64_t> convertedStaticTargetShape;
  SmallVector<Value> convertedDynamicTargetShape;
  dispatchIndexOpFoldResults(convertedTargetShape.value(),
                             convertedDynamicTargetShape,
                             convertedStaticTargetShape);
  return convertedDynamicTargetShape;
}

namespace {
/// Pattern to materialize the encoding for `hal.interface.binding.subspan`
/// operations.
struct MaterializeInterfaceBindingEncoding
    : public OpMaterializeEncodingPattern<
          IREE::HAL::InterfaceBindingSubspanOp> {
  using OpMaterializeEncodingPattern<
      IREE::HAL::InterfaceBindingSubspanOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "expected result type to be !flow.dispatch.tensor");
    }
    auto boundTensorType =
        llvm::dyn_cast<RankedTensorType>(resultType.getBoundType());
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "bound type is not a RankedTensorType");
    }

    auto convertedBoundType = getTypeConverter()->convertType(boundTensorType);
    if (convertedBoundType == boundTensorType) {
      return rewriter.notifyMatchFailure(subspanOp, "bound type already valid");
    }

    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    // Get the dynamic dims of the target.
    Location loc = subspanOp.getLoc();
    SmallVector<Value> newDynamicDims = subspanOp.getDynamicDims();
    FailureOr<SmallVector<Value>> convertedDynamicDims =
        getPackedDynamicDimsForDispatchTensor(
            rewriter, loc, *typeConverter, resultType,
            subspanOp.getDynamicDims(), this->materializeEncodingValueFn);
    // Drop the encoding if the target does not support it.
    if (succeeded(convertedDynamicDims)) {
      newDynamicDims = convertedDynamicDims.value();
    }

    auto newResultType = IREE::Flow::DispatchTensorType::get(
        resultType.getAccess(), convertedBoundType);
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getLayout(), subspanOp.getBinding(),
        subspanOp.getByteOffset(), newDynamicDims, subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorLoadOp
    : public OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorLoadOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial loads.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(loadOp, "bound type already valid");
    }

    Location loc = loadOp.getLoc();
    SmallVector<OpFoldResult> newMixedSizes = getMixedValues(
        boundTensorType.getShape(), loadOp.getSourceDims(), rewriter);
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                       sourceType, loadOp.getSourceDims(),
                                       this->materializeEncodingValueFn);
    if (succeeded(convertedMixedSizes)) {
      newMixedSizes = convertedMixedSizes.value();
    }
    SmallVector<OpFoldResult> newOffsets(newMixedSizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(newMixedSizes.size(),
                                         rewriter.getIndexAttr(1));
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        loadOp, adaptor.getSource(), newDynamicDims, newOffsets, newMixedSizes,
        newStrides);

    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorStoreOp
    : public OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorStoreOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial stores.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(storeOp, "bound type already valid");
    }

    Location loc = storeOp.getLoc();
    SmallVector<OpFoldResult> newMixedSizes = getMixedValues(
        boundTensorType.getShape(), storeOp.getTargetDims(), rewriter);
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                       targetType, storeOp.getTargetDims(),
                                       this->materializeEncodingValueFn);
    if (succeeded(convertedMixedSizes)) {
      newMixedSizes = convertedMixedSizes.value();
    }
    SmallVector<OpFoldResult> newOffsets(newMixedSizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(newMixedSizes.size(),
                                         rewriter.getIndexAttr(1));
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(), newDynamicDims,
        newOffsets, newMixedSizes, newStrides);
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Convert `set_encoding` op to `pack` op.
struct SetEncodingOpToPackOpConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::SetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::SetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    auto packOp = lowerSetEncodingOpToPackOp(rewriter, encodingOp,
                                             adaptor.getSource(), *converter,
                                             this->materializeEncodingValueFn);
    if (failed(packOp)) {
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          encodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(encodingOp, result);
      return success();
    }
    rewriter.replaceOp(encodingOp, packOp->getResult());
    return success();
  }
};

/// Convert `unset_encoding` op to `unpack` op.
struct UnsetEncodingOpToUnPackOpConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::UnsetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    auto unpackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, encodingOp, adaptor.getSource(), *converter,
        this->materializeEncodingValueFn);
    if (failed(unpackOp)) {
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          encodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(encodingOp, result);
      return success();
    }
    rewriter.replaceOp(encodingOp, unpackOp->getResult());
    return success();
  }
};

/// Generic pattern to convert operation that is in Destination Passing Style.
template <typename OpTy>
struct MaterializeDPSOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy dpsOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, dpsOp, adaptor.getInputs(), adaptor.getOutputs(), *converter,
        this->materializeEncodingValueFn);
    if (failed(convertedOp)) {
      return failure();
    }
    rewriter.replaceOp(dpsOp, convertedOp.value()->getResults());
    return success();
  }
};

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    FailureOr<Operation *> convertedOp =
        lowerOpWithEncoding(rewriter, op, adaptor.getOperands(), *converter,
                            this->materializeEncodingValueFn);
    if (failed(convertedOp))
      return failure();

    SmallVector<Value> replacements;
    for (auto [type, res] : llvm::zip_equal(
             op->getResultTypes(), convertedOp.value()->getResults())) {
      Type targetType = this->getTypeConverter()->convertType(type);
      replacements.push_back(
          rewriter.createOrFold<tensor::CastOp>(op.getLoc(), targetType, res));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct MaterializeOptimizationBarrierOp
    : public OpMaterializeEncodingPattern<IREE::Util::OptimizationBarrierOp> {
  using OpMaterializeEncodingPattern<
      IREE::Util::OptimizationBarrierOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Util::OptimizationBarrierOp op,
                  IREE::Util::OptimizationBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getOperandTypes(), [](Type type) -> bool {
          auto tensorType = dyn_cast<RankedTensorType>(type);
          return tensorType && tensorType.getEncoding();
        })) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Util::OptimizationBarrierOp>(
        op, adaptor.getOperands());
    return success();
  }
};

/// Pattern to convert contraction operations.
class MaterializeContractionOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
public:
  MaterializeContractionOp(
      MLIRContext *context,
      const MaterializeEncodingTypeConverter &typeConverter,
      MaterializeEncodingValueFn materializeEncodingValueFn = {},
      PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern<linalg::LinalgOp>(typeConverter, context,
                                                       benefit),
        materializeEncodingValueFn(materializeEncodingValueFn) {}

  LogicalResult
  matchAndRewrite(linalg::LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(op)) {
      return rewriter.notifyMatchFailure(
          op, "does not implement ContractionOpInterface");
    }

    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    FailureOr<Operation *> convertedOp =
        lowerContractionOpWithEncoding(rewriter, op, operands, *converter);
    if (failed(convertedOp)) {
      return failure();
    }
    rewriter.replaceOp(op.getOperation(), convertedOp.value()->getResult(0));
    return success();
  }

protected:
  const MaterializeEncodingValueFn materializeEncodingValueFn;
};

} // namespace

void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  MLIRContext *context = patterns.getContext();
  // TODO(hanchung): Move the generic op pattern to ShapeIndependent category
  // after we add the support for tile swizzling variants.
  patterns.insert<MaterializeDPSOperation<linalg::GenericOp>,
                  MaterializeContractionOp, SetEncodingOpToPackOpConversion,
                  UnsetEncodingOpToUnPackOpConversion>(
      context, typeConverter, materializeEncodingValueFn);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
}

void populateShapeIndependentMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  MLIRContext *context = patterns.getContext();
  typeConverter.addConversion(
      [&typeConverter](IREE::Flow::DispatchTensorType dispatchTensorType) {
        Type boundType = dispatchTensorType.getBoundType();
        Type convertedBoundType = typeConverter.convertType(boundType);
        if (convertedBoundType == boundType) {
          return dispatchTensorType;
        }
        return IREE::Flow::DispatchTensorType::get(
            dispatchTensorType.getAccess(), convertedBoundType);
      });

  target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
      [&typeConverter](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
        auto resultType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
            subspanOp.getResult().getType());
        // For types that are not `Flow::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });

  patterns.insert<
      MaterializeDPSOperation<linalg::FillOp>,
      MaterializeOperation<tensor::EmptyOp>, MaterializeOptimizationBarrierOp,
      MaterializeFlowDispatchTensorLoadOp, MaterializeFlowDispatchTensorStoreOp,
      MaterializeInterfaceBindingEncoding>(context, typeConverter,
                                           materializeEncodingValueFn);
};

} // namespace mlir::iree_compiler
