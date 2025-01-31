// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_FORMDISPATCHREGIONS_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_FORMDISPATCHREGIONS_H_

#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"

namespace mlir {

class Operation;

/// A rewriter that keeps track of all tensor::DimOps.
/// 用于跟踪，操作所有的DimOp
class TensorDimTrackingRewriter : public IRRewriter, IRRewriter::Listener {
public:
  /// Create a new rewriter: Scan the given op for tensor::DimOps.
  // 构造函数
  TensorDimTrackingRewriter(Operation *op);
  /// Return all tracked tensor::DimOps.
  SmallVector<tensor::DimOp> getTensorDimOps();

protected:
  void notifyOperationErased(Operation *op) override;
  void notifyOperationInserted(Operation *op, InsertPoint previous) override;

private:
  // 存储所有的DimOps，存储成operation指针。
  SmallPtrSet<Operation *, 16> dimOps;
};

} // namespace mlir

namespace mlir::iree_compiler::IREE::Flow {

/// Computes the workload and provides a workload region builder for the given
/// root op.
// 涉及到将操作分派到不同的执行单元，比如GPU的线程块或线程，需要根据操作的特点生成合适的工作负载配置。
FailureOr<IREE::Flow::WorkloadBuilder> getWorkloadBuilder(OpBuilder &builder,
                                                          Operation *rootOp);

/// Simplfy the given tensor::DimOps as much as possible.
/// * Static dimensions are replaced by constant.
/// * Dynamic dim ops are pushed as much as possible to the top of the function,
///   i.e., if the dim of a value is known to be equal to the dim of a value on
///   the reverse SSA use-def chain, rewrite the value with a dim op of that
///   value.
// 目的是尽可能简化tensor::DimOp。
// 静态维度会被替换为常量，动态维度的DimOp会被尽可能推到函数的顶部
// 比如如果某个值的维度与另一个值的维度相同，就替换为对那个值的DimOp。
// 这有助于减少运行时的计算，提升性能。
LogicalResult simplifyDimOps(RewriterBase &rewriter,
                             const SmallVector<tensor::DimOp> &dimOps);

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_FORMDISPATCHREGIONS_H_
