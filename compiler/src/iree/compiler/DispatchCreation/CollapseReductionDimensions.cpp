// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_COLLAPSEREDUCTIONDIMENSIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/**
collapseDimensions函数在这个代码段中的作用是对 linalg::LinalgOp操作的归约维度进行折叠处理，
目的是通过将多个维度合并成一个维度来简化操作，以便优化计算。
这种操作通常用于提高并行度或者减少计算的复杂度。
 */
/// Check whether the given dimensions are contiguous in the result map.
/// If non of the dimension are present in the map return true as well.
/// 这里的dims是reductio op的维度
static bool hasContiguousDims(AffineMap map, ArrayRef<unsigned> dims) {
  // 非投影操作，要么存在重复，要么存在交叉，肯定不是连续的
  if (!map.isProjectedPermutation())
    return false;
  llvm::SmallDenseSet<unsigned> existingDims(dims.begin(), dims.end());
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    if (map.getDimPosition(i) != dims[0]) {
      if (existingDims.count(map.getDimPosition(i))) {
        return false;
      }
      continue;
    }
    // Check that the following dimensions are match the order of `dims`
    for (unsigned j = 1, numDims = dims.size(); j < numDims; j++) {
      unsigned pos = i + j;
      if (pos >= map.getNumResults() || map.getDimPosition(pos) != dims[j]) {
        return false;
      }
    }
    break;
  }
  return true;
}

static SmallVector<ReassociationIndices>
collapseDimensions(linalg::LinalgOp linalgOp) {
  // ReassociationIndices: SmallVector<int64, 2>
  SmallVector<ReassociationIndices> collapseIndices;

  // ========= 判断是否满足折叠条件 ============
  // 判断该linalg::LinalgOp是否在一个dispatch region中
  // 或是一个dispatch workgroup中。
  if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
    return collapseIndices;
  }

  // reduction的维度给大于等于2，不然没有可折叠的维度
  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  if (reductionDims.size() < 2)
    return collapseIndices;

  // 判断affine map中的dim是否连续，只有affine map中的dimension是连续的
  // 才能够做dimension collapse
  for (AffineMap map : linalgOp.getIndexingMapsArray()) {
    if (!hasContiguousDims(map, reductionDims))
      return collapseIndices;
  }
  ReassociationIndices indices;
  for (unsigned dim : reductionDims) {
    indices.push_back(int64_t(dim));
  }
  collapseIndices.push_back(indices);
  return collapseIndices;
}

struct CollapseReductionDimensionsPass final
    : public impl::CollapseReductionDimensionsPassBase<
          CollapseReductionDimensionsPass> {
  // entry point
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // collapseDimensions是最重要改写pattern
    linalg::populateCollapseDimensions(patterns, collapseDimensions);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
