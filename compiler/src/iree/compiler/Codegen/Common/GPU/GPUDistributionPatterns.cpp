// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

namespace {

/// Given the state of the iterator, compute the indices of the original vector
/// that the current iterator state is iterating over. These indices are
/// parameterized by the thread grid.
static SmallVector<Value> computeSIMDIndex(const LayoutIterator::State &state,
                                           LayoutAttr layout, Value laneId,
                                           RewriterBase &rewriter) {
  MLIRContext *ctx = layout.getContext();
  AffineExpr threadX, threadY, threadZ;
  bindSymbols(ctx, threadX, threadY, threadZ);

  SmallVector<Value> simdIndex;
  // Calculate the index for each dim separately.
  for (PerDimLayoutAttr dimLayout : layout.getLayouts()) {
    AffineExpr offset = getAffineConstantExpr(0, ctx);
    AffineExpr stride = getAffineConstantExpr(1, ctx);
    for (auto [label, shape] : llvm::reverse(
             llvm::zip(dimLayout.getLabels(), dimLayout.getShapes()))) {
      int64_t position = state.lookup(label.getValue()).getPosition();

      switch (label.getValue()) {
      case LayoutDimension::LANEX:
        offset = offset + stride * threadX;
        break;
      case LayoutDimension::LANEY:
        offset = offset + stride * threadY;
        break;
      case LayoutDimension::LANEZ:
        offset = offset + stride * threadZ;
        break;
      default:
        offset = offset + stride * getAffineConstantExpr(position, ctx);
        break;
      }
      stride = stride * getAffineConstantExpr(shape, ctx);
    }

    auto [laneDimX, laneDimY, laneDimZ] = layout.getLaneGrid();
    SmallVector<Value> laneGrid = {
        rewriter.create<arith::ConstantIndexOp>(laneId.getLoc(), laneDimZ),
        rewriter.create<arith::ConstantIndexOp>(laneId.getLoc(), laneDimY),
        rewriter.create<arith::ConstantIndexOp>(laneId.getLoc(), laneDimX)};
    FailureOr<SmallVector<Value>> maybeReversedLaneGridVals =
        affine::delinearizeIndex(rewriter, laneId.getLoc(), laneId, laneGrid);
    assert(succeeded(maybeReversedLaneGridVals) &&
           "Failed to delinearize lane index");
    SmallVector<Value> laneGridVals = {(*maybeReversedLaneGridVals)[2],
                                       (*maybeReversedLaneGridVals)[1],
                                       (*maybeReversedLaneGridVals)[0]};

    // Compute the index for the dim.
    AffineMap indexMap = AffineMap::get(0, 3, offset);
    Value index = rewriter.create<affine::AffineApplyOp>(
        rewriter.getUnknownLoc(), indexMap, laneGridVals);
    simdIndex.push_back(index);
  }

  return simdIndex;
}

struct DistributeConstants final : OpDistributionPattern<arith::ConstantOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto constant = dyn_cast<VectorValue>(constantOp.getResult());
    if (!constant)
      return failure();

    // Only handle splat values for now.
    auto attr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();

    VectorLayoutInterface layout = signature[constant];

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType =
        VectorType::get(layout.getDistributedShape(), elementType);
    auto distributedOp = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), vectorType,
        SplatElementsAttr::get(vectorType, attr.getSplatValue<Attribute>()));
    replaceOpWithDistributedValues(rewriter, constantOp,
                                   distributedOp->getResult(0));
    return success();
  }
};

struct DistributeElementwise final
    : OpTraitDistributionPattern<OpTrait::Elementwise> {
  using OpTraitDistributionPattern::OpTraitDistributionPattern;

  LogicalResult matchAndRewrite(Operation *op, DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Check if this operation has elementwise mappable traits. This is
    // more restricted than only having elementwise trait.
    if (!OpTrait::hasElementwiseMappableTraits(op)) {
      return failure();
    }

    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      signature[vectorOperand]);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();

      // Distribute vector result types.
      if (auto vectorResult = dyn_cast<VectorValue>(result)) {
        VectorLayoutInterface resLayout = signature[vectorResult];
        resultType = VectorType::get(resLayout.getDistributedShape(),
                                     vectorResult.getType().getElementType());
      }
      resultTypes.push_back(resultType);
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp = mlir::clone(rewriter, op, resultTypes, operands);

    DistributionPattern::replaceOpWithDistributedValues(
        rewriter, op, distributedOp->getResults());
    return success();
  }
};

/// Given a projected permutation, get a reduced permutation, i.e. without
/// the projected dimensions.
static SmallVector<int64_t>
getReducedPermutation(AffineMap permutationMap,
                      llvm::SmallBitVector &unusedDims) {
  assert(permutationMap.isProjectedPermutation() &&
         "permutation map should be a projected permutation.");
  // TODO: The permutation map may also have broadcasting. Currently, we do not
  // handle it. This can be fixed by adding a "BROADCAST" dimension in the
  // layout.

  unusedDims.clear();
  unusedDims.resize(permutationMap.getNumDims(), true);

  for (AffineExpr dimExpr : permutationMap.getResults()) {
    int64_t pos = cast<AffineDimExpr>(dimExpr).getPosition();
    unusedDims[pos] = false;
  }

  SmallVector<int64_t> permutation;
  permutation.reserve(permutationMap.getNumResults());

  AffineMap reducedMap = compressUnusedDims(permutationMap);
  for (AffineExpr dimExpr : reducedMap.getResults()) {
    int64_t pos = cast<AffineDimExpr>(dimExpr).getPosition();
    permutation.push_back(pos);
  }

  return permutation;
}

template <typename OpTy>
struct DistributeXferLayoutAttr : OpDistributionPattern<OpTy> {
  static_assert(std::is_same<OpTy, vector::TransferReadOp>::value ||
                    std::is_same<OpTy, vector::TransferWriteOp>::value,
                "expected vector::TransferReadOp or vector::TransferWriteOp");

  DistributeXferLayoutAttr(MLIRContext *context, Value laneId,
                           PatternBenefit benefit = 1)
      : OpDistributionPattern<OpTy>(context, benefit), laneId(laneId) {}

  VectorValue accessMemory(OpTy xferOp, VectorValue accumulator,
                           LayoutAttr vectorLayout,
                           PatternRewriter &rewriter) const {
    // We need to take special consideration of the permutation map when
    // lowering. When accessing memory, we use the memoryLayout, because that
    // is how the data is accessed in memory. The data is stored in the vector
    // according to vectorLayout.
    llvm::SmallBitVector unusedDims;
    SmallVector<int64_t> permutation =
        getReducedPermutation(xferOp.getPermutationMap(), unusedDims);
    LayoutAttr memoryLayout =
        cast<LayoutAttr>(vectorLayout.permute(permutation));

    int loadWidth = getLoadStoreWidth(memoryLayout);
    DenseMap<LayoutDimension, int64_t> steps;
    steps[LayoutDimension::VECTORX] = loadWidth;
    LayoutIterator iterator(vectorLayout, steps);

    iterator.apply([&](const LayoutIterator::State &state) {
      SmallVector<Value> memoryIndices = getMemoryIndices(
          state, memoryLayout, xferOp.getIndices(), unusedDims, rewriter);
      SmallVector<int64_t> accIndices = state.computeSIMTIndex();
      accumulator = accessUnit(xferOp, memoryIndices, accIndices, accumulator,
                               vectorLayout, memoryLayout, rewriter);
    });

    return accumulator;
  }

  SmallVector<Value> getMemoryIndices(const LayoutIterator::State &state,
                                      LayoutAttr memoryLayout,
                                      SmallVector<Value> indices,
                                      llvm::SmallBitVector &projectedDims,
                                      RewriterBase &rewriter) const {
    SmallVector<Value> simdIndices =
        computeSIMDIndex(state, memoryLayout, laneId, rewriter);
    SmallVector<Value> memoryIndices(indices);

    // The memory layout has some projected leading dims that indices doesn't.
    int currSimd = 0;
    for (int i = 0, e = memoryIndices.size(); i < e; ++i) {
      if (projectedDims[i]) {
        continue;
      }

      memoryIndices[i] = rewriter.create<arith::AddIOp>(
          rewriter.getUnknownLoc(), memoryIndices[i], simdIndices[currSimd]);
      ++currSimd;
    }

    return memoryIndices;
  }

  virtual VectorValue accessUnit(OpTy xferOp, SmallVector<Value> &memoryIndices,
                                 SmallVector<int64_t> &accIndices,
                                 VectorValue accumulator,
                                 LayoutAttr vectorLayout,
                                 LayoutAttr memoryLayout,
                                 PatternRewriter &rewriter) const = 0;

  int getLoadStoreWidth(LayoutAttr layout) const {
    PerDimLayoutAttr fastestChanging = layout.getLayouts().back();
    if (std::optional<int64_t> width =
            fastestChanging.getShape(LayoutDimension::VECTORX)) {
      return *width;
    }
    return 1;
  }

  Value laneId;
};

struct DistributeTransferReadLayoutAttr final
    : DistributeXferLayoutAttr<vector::TransferReadOp> {
  using DistributeXferLayoutAttr::DistributeXferLayoutAttr;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    LayoutAttr vectorLayout =
        dyn_cast<LayoutAttr>(signature[readOp.getResult()]);
    if (!vectorLayout) {
      return failure();
    }

    // TODO: Return failure if we need masking.

    Type elementType = readOp.getSource().getType().getElementType();
    auto vectorType =
        VectorType::get(vectorLayout.getDistributedShape(), elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        readOp.getLoc(), vectorType, rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    VectorValue readVec = accessMemory(readOp, acc, vectorLayout, rewriter);

    replaceOpWithDistributedValues(rewriter, readOp, readVec);
    return success();
  }

  VectorValue accessUnit(vector::TransferReadOp readOp,
                         SmallVector<Value> &memoryIndices,
                         SmallVector<int64_t> &accIndices,
                         VectorValue accumulator, LayoutAttr vectorLayout,
                         LayoutAttr memoryLayout,
                         PatternRewriter &rewriter) const override {
    auto unitType = VectorType::get({getLoadStoreWidth(memoryLayout)},
                                    accumulator.getType().getElementType());
    VectorValue load = rewriter.create<vector::LoadOp>(
        readOp.getLoc(), unitType, readOp.getSource(), memoryIndices);
    return rewriter.create<vector::InsertStridedSliceOp>(
        readOp.getLoc(), load, accumulator, accIndices,
        SmallVector<int64_t>{1});
  }
};

struct DistributeTransferWriteLayoutAttr final
    : DistributeXferLayoutAttr<vector::TransferWriteOp> {
  using DistributeXferLayoutAttr::DistributeXferLayoutAttr;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    LayoutAttr vectorLayout =
        dyn_cast<LayoutAttr>(signature[writeOp.getVector()]);
    if (!vectorLayout) {
      return failure();
    }

    if (writeOp.getMask()) {
      return failure();
    }

    accessMemory(writeOp, writeOp.getVector(), vectorLayout, rewriter);

    rewriter.eraseOp(writeOp);
    return success();
  }

  VectorValue accessUnit(vector::TransferWriteOp writeOp,
                         SmallVector<Value> &memoryIndices,
                         SmallVector<int64_t> &accIndices,
                         VectorValue accumulator, LayoutAttr vectorLayout,
                         LayoutAttr memoryLayout,
                         PatternRewriter &rewriter) const override {
    int width = getLoadStoreWidth(memoryLayout);

    SmallVector<int64_t> strides(accIndices.size(), 1);
    SmallVector<int64_t> shapes(accIndices.size(), 1);
    shapes[shapes.size() - 1] = width;
    Value result = rewriter.create<vector::ExtractStridedSliceOp>(
        writeOp.getLoc(), getDistributed(rewriter, accumulator, vectorLayout),
        accIndices, shapes, strides);
    result = rewriter.create<vector::ExtractOp>(
        writeOp.getLoc(), result,
        SmallVector<int64_t>(accIndices.size() - 1, 0));
    rewriter.create<vector::StoreOp>(writeOp.getLoc(), result,
                                     writeOp.getSource(), memoryIndices);

    return accumulator;
  }
};

struct DistributeReductions final
    : OpDistributionPattern<vector::MultiDimReductionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeReductions(MLIRContext *context, int64_t maxBitsPerShuffle)
      : OpDistributionPattern(context), maxBitsPerShuffle(maxBitsPerShuffle) {}

  static constexpr int64_t kDefaultSubgroupSize = 32;

  // Do parallel reduction using butterfly shuffles.
  Value doThreadGlobalReduction(Value result, uint64_t shuffleOffset,
                                int64_t laneSize,
                                vector::CombiningKind combiningKind,
                                int64_t entriesPerVector, Value mEmpty,
                                OpBuilder &rewriter, Location loc) const {
    auto funcOp = result.getDefiningOp()->getParentOfType<func::FuncOp>();
    std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(funcOp);
    if (!maybeSubgroupSize)
      funcOp->emitWarning("No subgroup size specified, using default value = " +
                          Twine(kDefaultSubgroupSize));
    int64_t subgroupSize = maybeSubgroupSize.value_or(kDefaultSubgroupSize);

    Value mask;
    assert(llvm::isPowerOf2_64(laneSize));
    for (uint64_t i = shuffleOffset; i < shuffleOffset * laneSize; i <<= 1) {
      Value packed = packVectorToSupportedWidth(loc, rewriter, result);
      auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
          loc, packed, i, subgroupSize, gpu::ShuffleMode::XOR);
      Value unpacked =
          unpackToVector(loc, rewriter, shuffleOp.getShuffleResult(),
                         cast<VectorType>(result.getType()));
      result = makeArithReduction(rewriter, loc, combiningKind, unpacked,
                                  result, nullptr, mask);
    }

    // Reduce packed vector with initial value.
    Value reducedValue = rewriter.create<vector::ExtractOp>(
        loc, result, SmallVector<int64_t>{0});
    for (int i = 1; i < entriesPerVector; i++) {
      Value next = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{i});
      reducedValue = makeArithReduction(rewriter, loc, combiningKind,
                                        reducedValue, next, nullptr, mask);
    }
    result = makeArithReduction(rewriter, loc, combiningKind, reducedValue,
                                mEmpty, nullptr, mask);
    return result;
  }

  // This pattern distributes reductions as follows:
  // First, the data local to a specific thread is reduced.
  // Then, the data between threads is reduced by emitting appropriate
  // shuffle instructions.
  // Currently, only 16 and 32 bit types are supported.
  // TODO: Add ability to reduce n parallel dims together.
  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reductionOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> reductionDims = reductionOp.getReductionDims();
    // TODO: Add support for reductions along multiple dimensions.
    if (reductionDims.size() > 1)
      return failure();

    VectorValue resultVec = dyn_cast<VectorValue>(reductionOp.getResult());
    // TODO: Support results that are not vectors.
    if (!resultVec)
      return failure();
    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(signature[resultVec]);
    if (!resultLayout)
      return failure();

    VectorValue source = reductionOp.getSource();
    ShapedType sourceType = llvm::cast<ShapedType>(source.getType());
    // TODO: Add support for (n != 2)-D tensors.
    if (sourceType.getRank() != 2)
      return failure();

    LayoutAttr sourceLayout = dyn_cast<LayoutAttr>(signature[source]);
    if (!sourceLayout)
      return failure();

    VectorValue acc = dyn_cast<VectorValue>(reductionOp.getAcc());
    ShapedType accType = llvm::cast<ShapedType>(acc.getType());
    Type elementType = accType.getElementType();
    int bitWidth = elementType.getIntOrFloatBitWidth();
    // TODO: Support additional bitwidths.
    if ((bitWidth != 16) && (bitWidth != 32))
      return failure();

    Location loc = reductionOp.getLoc();
    auto storeVectorType =
        VectorType::get(resultLayout.getDistributedShape(), elementType);
    Value storeVec = rewriter.create<arith::ConstantOp>(
        loc, storeVectorType, rewriter.getZeroAttr(storeVectorType));

    int reductionDim = reductionDims[0];
    int parallelDim = reductionDim ^ 1;
    if (!sourceLayout.getLane(reductionDim))
      return failure();
    uint64_t shuffleOffset = sourceLayout.getShuffleOffset(reductionDim);
    int64_t laneSize = sourceLayout.getLaneDim(reductionDim).value();
    if (!llvm::isPowerOf2_64(laneSize))
      return failure();
    vector::CombiningKind combiningKind = reductionOp.getKind();

    auto reduceFn = [&](const LayoutIterator::State &state) {
      SmallVector<int64_t> parallelSimtIndices = state.computeSIMTIndex();
      Value mEmpty = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, acc, resultLayout),
          parallelSimtIndices);

      // Store one or more elements in packed vector depending on type.
      int64_t entriesPerVector = maxBitsPerShuffle / bitWidth;
      Value packedVector = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(
                   VectorType::get({entriesPerVector}, elementType)));

      int64_t index{0};
      Value result, mask;
      // Thread-local reduction.
      auto reduceLocalFn = [&](const LayoutIterator::State &state) {
        SmallVector<int64_t> indices = state.computeSIMTIndex();
        Value element = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, source, sourceLayout), indices);
        packedVector = rewriter.create<vector::InsertOp>(
            loc, element, packedVector, SmallVector<int64_t>{index});
        index = (index + 1) % entriesPerVector;
        // Reduce packed vector when full.
        if (index == 0) {
          result = result
                       ? makeArithReduction(rewriter, loc, combiningKind,
                                            result, packedVector, nullptr, mask)
                       : packedVector;
        }
      };

      LayoutIterator reductionIterator(sourceLayout, reductionDim);
      reductionIterator.maybeFreezeAndConcatenate(state);
      reductionIterator.apply(reduceLocalFn);

      // Thread-global reduction.
      result = doThreadGlobalReduction(result, shuffleOffset, laneSize,
                                       combiningKind, entriesPerVector, mEmpty,
                                       rewriter, loc);
      storeVec = rewriter.create<vector::InsertOp>(loc, result, storeVec,
                                                   parallelSimtIndices);
    };

    LayoutIterator parallelIterator(sourceLayout, parallelDim);
    parallelIterator.apply(reduceFn);
    replaceOpWithDistributedValues(rewriter, reductionOp, storeVec);

    return success();
  }

private:
  int64_t maxBitsPerShuffle;
};

struct DistributeScfFor final : OpDistributionPattern<scf::ForOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Block *oldLoopBody = forOp.getBody();

    // The new vector init_args of the loop.
    SmallVector<Value> newInitArgs;
    for (Value initArg : forOp.getInitArgs()) {
      if (auto vectorInitArg = dyn_cast<VectorValue>(initArg)) {
        if (isNonZeroRank(vectorInitArg)) {
          initArg =
              getDistributed(rewriter, vectorInitArg, signature[vectorInitArg]);
        }
      }
      newInitArgs.push_back(initArg);
    }

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block *loopBody = newForOp.getBody();

    // Set up new iter_args. The loop body uses SIMD, so wrap the SIMD iter_args
    // of the new loop op into ToSIMDOps.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> iterArgs = getBbArgsReplacements(
        rewriter, newForOp.getRegionIterArgs(), forOp.getInitArgs());
    iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

    // Move loop body to new loop.
    rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

    if (failed(distributeYield(rewriter, newForOp))) {
      return failure();
    }

    // Repleace loop results.
    replaceOpWithDistributedValues(rewriter, forOp, newForOp.getResults());
    return success();
  }

  LogicalResult distributeYield(PatternRewriter &rewriter,
                                scf::ForOp forOp) const {
    scf::YieldOp yieldOp =
        llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    std::optional<DistributionSignature> maybeSignature =
        getOpSignature(yieldOp);
    if (!maybeSignature) {
      return failure();
    }
    DistributionSignature signature = *maybeSignature;

    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : yieldOp->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        // Distributing the operand requires it to have a non-zero rank, meaning
        // it must have at least one dimension. If the vector has a non-zero
        // rank, the operand is distributed according to the provided layout
        // signature.
        if (isNonZeroRank(vectorOperand)) {
          operand = DistributionPattern::getDistributed(
              rewriter, vectorOperand, signature[vectorOperand]);
        }
      }
      operands.push_back(operand);
    }

    // Since this operation has no results, we can directly replace it using
    // the standard API.
    auto distributedYieldOp =
        rewriter.create<scf::YieldOp>(yieldOp.getLoc(), operands);
    rewriter.replaceOp(yieldOp, distributedYieldOp);
    return success();
  }

  /// Helper function for loop distribution. Given a list of bbArgs of the new
  /// (distributed) loop op, wrap the distributed vector args (now distributed)
  /// into ToSIMDOps, so that the block body can be moved over to the new op.
  SmallVector<Value> getBbArgsReplacements(RewriterBase &rewriter,
                                           Block::BlockArgListType bbArgs,
                                           ValueRange oldInits) const {
    SmallVector<Value> replacements;
    for (auto [bbArg, oldInit] : llvm::zip_equal(bbArgs, oldInits)) {
      Value val = bbArg;
      if (auto oldVectorInit = dyn_cast<VectorValue>(oldInit)) {
        if (isNonZeroRank(oldVectorInit)) {
          val = rewriter.create<IREE::VectorExt::ToSIMDOp>(
              oldVectorInit.getLoc(), oldVectorInit.getType(), val);
        }
      }
      replacements.push_back(val);
    }
    return replacements;
  }
};

struct DistributeTransposeLayoutAttr final
    : OpDistributionPattern<vector::TransposeOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue value = transposeOp.getVector();
    VectorLayoutInterface layout = dyn_cast<LayoutAttr>(signature[value]);
    if (!layout) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "layout must be LayoutAttr");
    }

    /// Transpose only changes the notion of where the data carried by each
    /// thread comes from in the SIMD vector. The data carried by each thread is
    /// still the same, just iterated in a new permuted order. This iteration
    /// information is carried by the layout. So, we can simply distribute
    /// transpose to a no-op. Example:
    ///
    /// input: vector<2x4xf16>
    ///
    /// 0 0 1 1
    /// 2 2 3 3
    ///
    /// after transpose,
    ///
    /// transp: vector<4x2xf16>
    ///
    /// 0 2
    /// 0 2
    /// 1 3
    /// 1 3
    ///
    /// As it can be seen, each thread is still carrying the same data and
    /// distributes to vector<2xf16>.
    ///
    /// The only difference is where this vector<2xf16> comes from and that
    /// before transpose, this vector<2xf16> was representing the fastest
    /// changing dimension, but after distribution it's not.
    replaceOpWithDistributedValues(rewriter, transposeOp,
                                   getDistributed(rewriter, value, layout));
    return success();
  }
};

struct DistributeBroadcastLayoutAttr final
    : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {

    VectorValue source = dyn_cast<VectorValue>(broadcastOp.getSource());
    if (!source) {
      // TODO: Add support for scalar broadcasting.
      return failure();
    }

    VectorValue vector = broadcastOp.getVector();
    LayoutAttr layout = dyn_cast<LayoutAttr>(signature[vector]);
    if (!layout) {
      return failure();
    }

    VectorLayoutInterface sourceLayout = signature[source];

    // We currently only support 1-D to 2-D broadcasting.
    if (source.getType().getRank() != 1 || vector.getType().getRank() != 2) {
      return failure();
    }

    int broadcastedDim = 0;
    int parallelDim = 1;

    Type elementType =
        llvm::cast<ShapedType>(vector.getType()).getElementType();
    auto vectorType =
        VectorType::get(layout.getDistributedShape(), elementType);
    Location loc = broadcastOp.getLoc();
    Value accumulator = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    // Iterate over the parallel dimension.;
    LayoutIterator parallelIterator(layout, parallelDim);
    parallelIterator.apply([&](const LayoutIterator::State &parallelState) {
      // Extract the value from source.
      SmallVector<int64_t> sourceIndices = parallelState.computeSIMTIndex();
      Value value = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, source, sourceLayout), sourceIndices);

      // Broadcast value over the broadcasted dimension.
      LayoutIterator broadcastIterator(layout, broadcastedDim);
      broadcastIterator.maybeFreezeAndConcatenate(parallelState);
      broadcastIterator.apply([&](const LayoutIterator::State &broadcastState) {
        SmallVector<int64_t> resultIndices = broadcastState.computeSIMTIndex();

        accumulator = rewriter.create<vector::InsertOp>(loc, value, accumulator,
                                                        resultIndices);
      });
    });

    replaceOpWithDistributedValues(rewriter, broadcastOp, accumulator);
    return success();
  }
};

/// This pattern implements a distribution pattern for layout conflict
/// resolutions where the resolution is a simple vector reshape.
/// In most cases, layout conflicts will need to be resolved with a
/// trip to shared memory or shuffle instructions and in those scenarios
/// this pattern will not work.
///
/// Below we outline some scenarios where this pattern will be useful:
/// - Unary Operators which are permutation invariant
///   Example:
///     Say the data for a single row is distributed among 2 threads as
///     0 0 0 0 1 1 1 1
///     and we have a layout conflict that requires the data to be
///     distributed as
///     0 0 1 1 0 0 1 1
///     and we are interested in computing an elementwise operation like exp
///     or trying to do a reduction along the row, then since the operations
///     are permutation invariant, we can treat the resolution as a vector
///     reshape.
/// - Binary Operators which are permutation invariant
///   Example:
///     Using the same example as above, say we are trying to do a dot product
///     between two vectors that have the above layout. As long as both
///     operands are permuted the same way, we will end up with the correct
///     sequence of multiplications and additions.
///
struct DistributeLayoutConflictResolutions final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  VectorValue reshapeVector(Location loc, RewriterBase &rewriter,
                            VectorValue src, LayoutAttr &currentLayout,
                            LayoutAttr &targetLayout, Type elementType) const {

    SmallVector<int64_t> targetShape = targetLayout.getDistributedShape();
    SmallVector<int64_t> currentShape = currentLayout.getDistributedShape();

    auto newVectorType = VectorType::get(targetShape, elementType);
    auto constantOp = rewriter.create<arith::ConstantOp>(
        loc, newVectorType, rewriter.getZeroAttr(newVectorType));
    auto newVector = dyn_cast<VectorValue>(constantOp.getResult());

    int64_t innermostDim = targetShape.size() - 1;
    int64_t step =
        std::min(targetShape[innermostDim], currentShape[innermostDim]);
    DenseMap<LayoutDimension, int64_t> steps;
    LayoutDimension vecDim = LayoutDimension::VECTORX;
    steps[vecDim] = step;
    LayoutIterator srcIterator(currentLayout, steps);
    LayoutIterator targetIterator(targetLayout, steps);

    for (; !srcIterator.iterationComplete() &&
           !targetIterator.iterationComplete();
         ++srcIterator, ++targetIterator) {
      SmallVector<int64_t> srcOffset =
          srcIterator.getState().computeSIMTIndex();
      SmallVector<int64_t> targetOffset =
          targetIterator.getState().computeSIMTIndex();
      SmallVector<int64_t> sliceSize(srcOffset.size(), 1);
      sliceSize[sliceSize.size() - 1] = step;
      SmallVector<int64_t> sliceStride(srcOffset.size(), 1);
      Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, src, srcOffset, sliceSize, sliceStride);
      newVector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, slice, newVector, targetOffset, sliceStride);
    }
    return newVector;
  }

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp resolutionOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto vector = cast<VectorValue>(resolutionOp.getInput());
    auto result = cast<VectorValue>(resolutionOp.getOutput());
    LayoutAttr currentLayout = dyn_cast<LayoutAttr>(signature[vector]);
    if (!currentLayout)
      return failure();
    LayoutAttr targetLayout = dyn_cast<LayoutAttr>(signature[result]);
    if (!targetLayout)
      return failure();

    if (currentLayout == targetLayout) {
      return rewriter.notifyMatchFailure(
          resolutionOp, "Layout conversion is not a conflict.");
    }

    SmallVector<int64_t> currentVecShape = currentLayout.getDistributedShape();
    SmallVector<int64_t> targetVecShape = targetLayout.getDistributedShape();
    if (currentVecShape.size() != targetVecShape.size())
      return failure();

    auto numElements = [](ArrayRef<int64_t> vector) {
      return std::accumulate(vector.begin(), vector.end(), 1,
                             std::multiplies<int64_t>());
    };
    if (numElements(currentVecShape) != numElements(targetVecShape))
      return failure();

    if (currentLayout.hasLaneConflictWith(targetLayout)) {
      return failure();
    }

    Type elementType =
        llvm::cast<VectorType>(result.getType()).getElementType();
    Value newVector =
        reshapeVector(resolutionOp.getLoc(), rewriter,
                      getDistributed(rewriter, vector, targetLayout),
                      currentLayout, targetLayout, elementType);
    replaceOpWithDistributedValues(rewriter, resolutionOp, newVector);
    return success();
  }
};

/// Pattern that allows us to write to shared memory
/// and read back to register with correct layouts.
/// especially used when we don't have an optimized way
/// to resolve the conflict.
struct DistributeLayoutConflictToSharedMemory final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp resolutionOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto loc = resolutionOp.getLoc();
    auto vector = cast<VectorValue>(resolutionOp.getInput());
    auto result = cast<VectorValue>(resolutionOp.getOutput());
    LayoutAttr currentLayout = dyn_cast<LayoutAttr>(signature[vector]);
    if (!currentLayout) {
      return rewriter.notifyMatchFailure(resolutionOp,
                                         "Source layout must be LayoutAttr.");
    }
    LayoutAttr targetLayout = dyn_cast<LayoutAttr>(signature[result]);
    if (!targetLayout) {
      return rewriter.notifyMatchFailure(resolutionOp,
                                         "Target layout must be LayoutAttr.");
    }

    if (currentLayout == targetLayout) {
      return rewriter.notifyMatchFailure(
          resolutionOp, "Layout conversion is not a conflict.");
    }

    SmallVector<int64_t> currentVecShape = currentLayout.getDistributedShape();
    SmallVector<int64_t> targetVecShape = targetLayout.getDistributedShape();
    if (currentVecShape.size() != targetVecShape.size()) {
      return rewriter.notifyMatchFailure(
          resolutionOp,
          "Target's and source's distributed rank needs to match.");
    }

    auto numElements = [](ArrayRef<int64_t> vector) {
      return std::accumulate(vector.begin(), vector.end(), 1,
                             std::multiplies<int64_t>());
    };

    if (numElements(currentVecShape) == numElements(targetVecShape) &&
        !currentLayout.hasLaneConflictWith(targetLayout)) {
      // If the conditions suffice, we can skip the trip to shared memory
      // and just use the default/more efficient layout conflict resolution
      // distribution.
      return rewriter.notifyMatchFailure(resolutionOp,
                                         "Failing because condition suffice to "
                                         "use better conflict resolutions.");
    }

    // Compute Subgroup and Workgroup related information and offsets.
    auto funcOp = resolutionOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          resolutionOp, "Expects a parent of type funcOp S.T we can compute "
                        "subgroup and workgroup related information.");
    }
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!workgroupSize.has_value() || !subgroupSize.has_value()) {
      return rewriter.notifyMatchFailure(
          resolutionOp, "Expects workgroup/subgroup information to be "
                        "available to resolve conflict.");
    }
    int64_t flatThreadSize = ShapedType::getNumElements(workgroupSize.value());
    if (flatThreadSize % subgroupSize.value() != 0)
      return failure();
    int64_t numSubgroups = flatThreadSize / subgroupSize.value();

    // Define shapes and types needed to be roundtripped to shared-memory.
    // The allocated shared-memory will stack subgroup data
    // on fastest dimension. Hence, shape will be:
    // [dim0, dim1, ..., subgroupCount * dimN]

    auto resolutionType =
        llvm::dyn_cast_or_null<VectorType>(resolutionOp.getResult().getType());
    if (!resolutionType) {
      return rewriter.notifyMatchFailure(
          resolutionOp,
          "Expects resolutionOp result to be of type vectorType.");
    }
    if (!resolutionType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          resolutionOp, "Expects resolutionOp result to have static shape.");
    }
    auto paddedShape = SmallVector<int64_t>(resolutionType.getShape());
    int64_t vectorRank = resolutionType.getRank();
    paddedShape[vectorRank - 1] *= numSubgroups;

    // Offset and indexing computation such that subgroups can
    // write and read to shared memory correctly and without conflicts.
    AffineExpr d0, d1, d2, s0;
    bindDims(rewriter.getContext(), d0, d1, d2);
    bindSymbols(rewriter.getContext(), s0);
    auto indexType = rewriter.getIndexType();
    Value threadX =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x);
    Value threadY =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::y);
    Value threadZ =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::z);
    Value flatThreadId = affine::makeComposedAffineApply(
        rewriter, loc,
        (d0 + workgroupSize.value()[0] * d1 +
         (workgroupSize.value()[0] * workgroupSize.value()[1]) * d2),
        {threadX, threadY, threadZ});
    Value subgroupOffset = affine::makeComposedAffineApply(
        rewriter, loc,
        s0.floorDiv(subgroupSize.value()) *
            resolutionType.getShape()[vectorRank - 1],
        {flatThreadId});

    // Create shared memory to store the intermediate from src layout.
    auto workgroupMemoryAddressSpace = Attribute(gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::AddressSpace::Workgroup));
    MemRefType allocType =
        MemRefType::get(paddedShape, resolutionType.getElementType(),
                        AffineMap(), workgroupMemoryAddressSpace);
    auto alloc = rewriter.create<memref::AllocOp>(loc, allocType);

    SmallVector<OpFoldResult> offsets(vectorRank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(vectorRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> shapes = llvm::to_vector(
        llvm::map_range(resolutionType.getShape(), [&](int64_t dim) {
          return OpFoldResult(rewriter.getIndexAttr(dim));
        }));
    offsets[vectorRank - 1] = subgroupOffset;
    auto subview = rewriter.create<memref::SubViewOp>(loc, alloc, offsets,
                                                      shapes, strides);

    // Creating write/trip to shared memory using src layout.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(resolutionType.getRank(), c0);
    SmallVector<bool> inBounds(vectorRank, true);
    auto write = rewriter.create<vector::TransferWriteOp>(loc, vector, subview,
                                                          indices, inBounds);
    // Insert gpu.barrier
    rewriter.create<gpu::BarrierOp>(write.getLoc());

    // Creating read from shared memory using dst layout.
    // Read with offset starting from the warpIdx * OG fastest dim.
    indices[vectorRank - 1] = subgroupOffset;
    auto read = rewriter.create<vector::TransferReadOp>(loc, resolutionType,
                                                        alloc, indices);

    // Set layouts signature for write.
    // We need to set the layout on the srcVector/first operand.
    auto unitAttr = UnitAttr::get(rewriter.getContext());
    auto writeAttrs = SmallVector<Attribute>(write->getNumOperands(), unitAttr);
    writeAttrs[0] =
        currentLayout; // 1st operand is src which requires currentLayout.
    ArrayAttr writeOperandsAttr =
        ArrayAttr::get(rewriter.getContext(), writeAttrs);
    ArrayAttr writeResultsAttr = ArrayAttr::get(rewriter.getContext(), {});
    setSignatureForRedistribution(rewriter, write.getOperation(),
                                  writeOperandsAttr, writeResultsAttr);

    // Set layouts signature for read.
    // We only need to set the layout on output.
    ArrayAttr readOperandsAttr = ArrayAttr::get(
        rewriter.getContext(),
        SmallVector<Attribute>(read->getNumOperands(), unitAttr));
    ArrayAttr readResultsAttr =
        ArrayAttr::get(rewriter.getContext(), {targetLayout});
    setSignatureForRedistribution(rewriter, read.getOperation(),
                                  readOperandsAttr, readResultsAttr);

    rewriter.replaceOp(resolutionOp, read.getResult());
    return success();
  }
};

struct DistributeTrivialLayoutConversions final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto input = cast<VectorValue>(toLayoutOp.getInput());
    auto output = cast<VectorValue>(toLayoutOp.getOutput());
    VectorLayoutInterface currentLayout = signature[input];
    VectorLayoutInterface targetLayout = signature[output];

    if (!currentLayout) {
      return rewriter.notifyMatchFailure(toLayoutOp, "No layout set on input");
    }

    if (!targetLayout) {
      return rewriter.notifyMatchFailure(toLayoutOp, "No layout set on output");
    }

    if (currentLayout != targetLayout) {
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "Non-trivial layout conversion.");
    }

    rewriter.replaceOp(toLayoutOp, toLayoutOp.getOperand());
    return success();
  }
};

struct DistributeGather final : OpDistributionPattern<vector::GatherOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::GatherOp gatherOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = gatherOp.getResult();
    VectorValue indexVec = gatherOp.getIndexVec();
    VectorValue mask = gatherOp.getMask();
    VectorValue passThru = gatherOp.getPassThru();

    VectorLayoutInterface resultLayout = signature[result];
    VectorLayoutInterface indicesLayout = signature[indexVec];
    VectorLayoutInterface maskLayout = signature[mask];
    VectorLayoutInterface passThruLayout = signature[passThru];

    if (!resultLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "result does not have layout");
    }
    if (!indicesLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "indices does not have layout");
    }
    if (!maskLayout) {
      return rewriter.notifyMatchFailure(gatherOp, "mask does not have layout");
    }
    if (!passThruLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "passThru does not have layout");
    }

    SmallVector<int64_t> distributedShape = resultLayout.getDistributedShape();
    Type elementType = result.getType().getElementType();
    VectorType distributedType = VectorType::get(distributedShape, elementType);

    // Simply distribute all operands and results.
    VectorValue distributed = rewriter.create<vector::GatherOp>(
        gatherOp.getLoc(), distributedType, gatherOp.getBase(),
        gatherOp.getIndices(),
        getDistributed(rewriter, indexVec, indicesLayout),
        getDistributed(rewriter, mask, maskLayout),
        getDistributed(rewriter, passThru, passThruLayout));

    replaceOpWithDistributedValues(rewriter, gatherOp, distributed);
    return success();
  }
};

} // namespace

void populateGPUReductionDistributionPatterns(RewritePatternSet &patterns,
                                              int64_t maxBitsPerShuffle) {
  patterns.add<DistributeReductions>(patterns.getContext(), maxBitsPerShuffle);
}

void populateGPUDistributionPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributeConstants, DistributeScfFor>(patterns.getContext());
  // Elementwise patterns.
  patterns.add<DistributeElementwise>(patterns.getContext());
  patterns.add<DistributeTrivialLayoutConversions>(patterns.getContext());
  // Gather patterns.
  patterns.add<DistributeGather>(patterns.getContext());
}

void populateGPUDistributionLayoutAttrPatterns(Value laneId,
                                               RewritePatternSet &patterns) {
  patterns
      .add<DistributeTransferReadLayoutAttr, DistributeTransferWriteLayoutAttr>(
          patterns.getContext(), laneId);
  patterns.add<DistributeBroadcastLayoutAttr, DistributeTransposeLayoutAttr>(
      patterns.getContext());
}

// TODO: Need a new op/analysis to determine when this pattern is safe to use.
void populateGPULayoutResolutionDistributionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DistributeLayoutConflictResolutions,
               DistributeLayoutConflictToSharedMemory>(patterns.getContext());
}

}; // namespace mlir::iree_compiler
