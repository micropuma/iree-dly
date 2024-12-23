// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <numeric>

namespace mlir::iree_compiler {

using IREE::Encoding::EncodingAttr;
using IREE::Encoding::getEncodingAttr;
using IREE::Encoding::getEncodingContractionDims;

// If tensorType has the encoding of a matmul RESULT with narrow N, returns
// the transposed type. Otherwise, just returns tensorType.
static RankedTensorType transposeIfNarrowNResult(RankedTensorType tensorType) {
  auto encoding =
      llvm::dyn_cast_or_null<EncodingAttr>(tensorType.getEncoding());
  if (!encoding) {
    return tensorType;
  }
  if (!isNarrowNResult(encoding)) {
    return tensorType;
  }
  SmallVector<int64_t> newOriginalShape(tensorType.getShape());
  auto userIndexingMaps = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> maps;
  for (auto a : userIndexingMaps) {
    maps.push_back(cast<AffineMapAttr>(a).getAffineMap());
  }
  auto cDims = linalg::inferContractionDims(maps);
  SmallVector<int64_t> newShape(tensorType.getShape());
  SmallVector<int64_t> permIndices(maps[0].getNumDims());
  std::iota(std::begin(permIndices), std::end(permIndices), 0);
  // Matrix case: there are both M and N dimensions. Transposing means swapping
  // them.
  if (cDims->m.size() == 1 && cDims->n.size() == 1) {
    int m = cDims->m[0];
    int n = cDims->n[0];
    std::swap(permIndices[m], permIndices[n]);
    std::optional<unsigned> mDim = encoding.mapDimToOperandIndex(m);
    std::optional<unsigned> nDim = encoding.mapDimToOperandIndex(n);
    if (mDim.has_value() && nDim.has_value()) {
      std::swap(newShape[mDim.value()], newShape[nDim.value()]);
      std::swap(newOriginalShape[mDim.value()], newOriginalShape[nDim.value()]);
    }
  }
  // Vector case: there is no N dimension to swap the M dimension with. We
  // swap the maps themselves.
  if (cDims->n.empty()) {
    std::swap(maps[0], maps[1]);
  }

  SmallVector<int64_t> newRoundDimsTo(encoding.getRoundDimsToArray());
  assert(newRoundDimsTo.size() == 0 || newRoundDimsTo.size() >= 3);
  if (newRoundDimsTo.size() != 0) {
    std::swap(newRoundDimsTo[newRoundDimsTo.size() - 3],
              newRoundDimsTo[newRoundDimsTo.size() - 2]);
  }
  auto context = tensorType.getContext();
  AffineMap permutation = AffineMap::getPermutationMap(permIndices, context);
  for (auto &map : maps) {
    map = map.compose(permutation);
  }
  auto elemType = tensorType.getElementType();
  auto operandIndex = encoding.getOperandIndex().getInt();

  // TODO(#17718): Handle the broadcast map for transpose cases. It is on the
  // experimental path, so it is not clear what needs to be done here. For now
  // just use the original map for the new encoding.
  std::optional<AffineMap> newBcastMap;
  if (encoding.getBcastMap()) {
    newBcastMap = encoding.getBcastMap().getValue();
  }
  auto newEncoding = IREE::Encoding::EncodingAttr::get(
      context, operandIndex, encoding.getOpType().getValue(),
      encoding.getElementTypesArray(), maps, newBcastMap, newRoundDimsTo);
  return RankedTensorType::get(newShape, elemType, newEncoding);
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    MaterializeEncodingFn materializeEncodingFn,
    IREE::HAL::ExecutableTargetAttr targetAttr, bool transposeNarrowN)
    : materializeEncodingFn(materializeEncodingFn), targetAttr(targetAttr),
      transposeNarrowN(transposeNarrowN) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType type) -> RankedTensorType {
    // For a given tensor type with an encoding, return the materialized
    // type to use for it. If no encoding is set, then return the tensor type
    // itself.
    RankedTensorType tensorType =
        transposeNarrowN ? transposeIfNarrowNResult(type) : type;
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfo(tensorType);
    if (failed(maybeEncodingInfo)) {
      return dropEncoding(type);
    }
    auto encodingInfo = *maybeEncodingInfo;
    auto packedType = cast<RankedTensorType>(tensor::PackOp::inferPackedType(
        tensorType, maybeEncodingInfo->innerTileSizes,
        maybeEncodingInfo->innerDimsPos, maybeEncodingInfo->outerDimsPerm));

    // There is no swizzle, we are already done. Typically the case on CPU.
    if (!encodingInfo.swizzle) {
      return packedType;
    }

    // There is a swizzle, we need to handle it. Typically the case on GPU.
    auto swizzle = *encodingInfo.swizzle;
    SmallVector<int64_t> newShape(
        packedType.getShape().drop_back(encodingInfo.innerTileSizes.size()));
    SmallVector<int64_t> swizzledTileShape =
        getExpandedTileShape(swizzle.expandShape);
    applyPermutationToVector(swizzledTileShape, swizzle.permutation);
    newShape.append(swizzledTileShape);
    return RankedTensorType::get(newShape, packedType.getElementType());
  });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasEncoding = [](Type t) -> bool {
      auto tensorType = dyn_cast<RankedTensorType>(t);
      return tensorType && tensorType.getEncoding();
    };
    auto valueHasEncoding = [=](Value v) -> bool {
      return typeHasEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

RankedTensorType dropEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingAttr encoding,
                                                 int64_t rank,
                                                 TileMxNxK tileMxNxK) {
  MaterializeEncodingInfo encodingInfo;
  auto cDims = getEncodingContractionDims(encoding);
  // The following expects M, N, K, and Batch sizes of at most 1 for now
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() == 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  std::optional<unsigned> batchDim =
      cDims->batch.empty() ? std::nullopt
                           : encoding.mapDimToOperandIndex(cDims->batch[0]);
  std::optional<unsigned> mDim =
      cDims->m.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->m[0]);
  std::optional<unsigned> nDim =
      cDims->n.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims->k[0]);
  if (batchDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(batchDim.value());
  }
  if (mDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(mDim.value());
    encodingInfo.innerDimsPos.push_back(mDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (nDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(nDim.value());
    encodingInfo.innerDimsPos.push_back(nDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (kDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(kDim.value());
    encodingInfo.innerDimsPos.push_back(kDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

bool isNarrowNResult(EncodingAttr encoding) {
  if (encoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RESULT) {
    return false;
  }

  return IREE::Encoding::getMatmulNarrowDim(encoding).isN();
}

SmallVector<int64_t>
getExpandedTileShape(const TileSwizzle::ExpandShapeType &expandShape) {
  SmallVector<int64_t> result;
  for (auto e : expandShape) {
    for (auto d : e) {
      result.push_back(d.size);
    }
  }
  return result;
}

} // namespace mlir::iree_compiler
