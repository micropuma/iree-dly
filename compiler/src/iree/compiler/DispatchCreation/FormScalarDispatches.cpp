// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-dispatch-creation-form-scalar-dispatches"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FORMSCALARDISPATCHESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Pass declaration.
struct FormScalarDispatchesPass final
    : public impl::FormScalarDispatchesPassBase<FormScalarDispatchesPass> {
  void runOnOperation() override;
};
} // namespace

/// Return true if type represents a value less than `n` elements.
/// 辅助函数
static bool isScalarOrTensorOfLinearSizeN(int n, Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return true;
  }

  // 重点是就是针对tensor op，
  // 需要通过判断tensor op的element的数量。
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    // dispatch流程不支持动态shape
    if (!tensorType.hasStaticShape()) {
      return false;
    }
    return tensorType.getNumElements() <= n;
  }
  return false;
}

/// 标记compute operations
/// Return `true` for operations that are to be treated as compute operations.
static bool isComputeOperation(Operation *op) {
  MLIRContext *context = op->getContext();

  // 核心逻辑是，所有的LinalgDialect都是computeOperation
  // TensorDialect中，除了个别的operation，均是computeOperation。
  // 比如tensor::CollapseShapeOp就不是computeOperation
  if (op->getDialect() == context->getLoadedDialect<linalg::LinalgDialect>()) {
    return true;
  }
  if (op->getDialect() == context->getLoadedDialect<tensor::TensorDialect>()) {
    return !isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::EmptyOp,
                tensor::ExpandShapeOp, tensor::PackOp, tensor::UnPackOp>(op);
  }
  return false;
}

/// 判断operation的workload的大小
/// Return `true` if the workload of this operation is less than `n`.
static bool isOperationWorkloadLessThanSizeN(int n, Operation *candidateOp) {
  return llvm::all_of(candidateOp->getOperands(),
                      [&](Value v) {
                        return isScalarOrTensorOfLinearSizeN(n, v.getType());
                      }) &&
         llvm::all_of(candidateOp->getResultTypes(), [&](Type t) {
           return isScalarOrTensorOfLinearSizeN(n, t);
         });
}

/// Return `true` is the operation is to be treated as a scalar operation
/// and moved into a scalar dispatch (not necessarily as the root of the
/// dispatch).
/// 虽然名字叫scalar operation，但其实tensor of restricted size也是可以的
static bool isScalarOperation(int workload, Operation *op) {
  // 1. Ignore most operations. Only look for a whitelist set of operations.
  if (!isComputeOperation(op)) {
    return false;
  }

  // 2. Check that the workload of the operation is less then the limit
  if (!isOperationWorkloadLessThanSizeN(workload, op)) {
    return false;
  }

  // 3. Do not move operations that are cloned into the dispatch region.
  // TODO: This might prevent moving all scalar operations into dispatch
  // resulting in artifical splits. Revisit after more examples.
  // 根据函数的定义，似乎这条语句的逻辑是找到可以clone的函数
  return !IREE::Flow::isClonableIntoDispatchOp(op);
}

/// Given a `rootOp` return a DAG of the program that represents
/// operations that can be moved into a scalar dispatch with the `rootOp`
/// as the root of the DAG.
/// 这个函数是整个ScalarDispatch的核心，对于给定的rootOp，
/// 用一个map存储所有和该rootOp划分到同一个dispatch region的operations
llvm::SetVector<Operation *> computeSliceToMoveIntoDispatch(
    int workload, Operation *rootOp,
    const llvm::DenseMap<Operation *, Operation *> &opToRootMap) {
  BackwardSliceOptions options;

  // 定义filter辅助函数
  // 该filter有如下过滤条件：
  // 1. 必须是scalar op（基础op + tensor<1> op）
  // 2. 该op的所有user已经放入rootOp的dispatch候选
  // 3. 该op必须和rootOp在同一block，不可跨越block
  // 4. 该op还没有加入任何一个dispatch
  options.filter = [&](Operation *currentOp) {
    assert(currentOp && "current op is null");
    if (opToRootMap.count(currentOp)) {
      return false;
    }
    // Operations needs to be in the same block as `rootOp`.
    if (currentOp->getBlock() != rootOp->getBlock()) {
      return false;
    }

    if (!isScalarOperation(workload, currentOp)) {
      return false;
    }

    // All its uses must be in the `opToRootMap`, i.e. they are either
    // in the current dispatches, or those already formed.
    return llvm::all_of(currentOp->getUsers(), [&](Operation *user) {
      return opToRootMap.count(user);
    });
  };
  options.omitBlockArguments = true;
  llvm::SetVector<Operation *> slice;

  // 基于先前定义好的filter，对于currentOp做backward slice analysis
  getBackwardSlice(rootOp, &slice, options);
  return slice;
}

/// Return `true` if the op is to be treated as a root of a scalar dispatch.
/// 这个pass的核心部分是找寻到哪个operation可以作为dispatch region的root来使用。
static bool isSliceRoot(int workload, Operation *op) {
  // 一个operation的父亲不是DispatchRegionOp，并且是scalar op
  return !op->getParentOfType<IREE::Flow::DispatchRegionOp>() &&
         isScalarOperation(workload, op);
}

// Form dispatch regions from slice of the operation.
// 基于分析出来的rootOp，构建Flow::DispatchRegionOp
static FailureOr<IREE::Flow::DispatchRegionOp>
formDispatchRegionFromSlice(RewriterBase &rewriter, Operation *rootOp,
                            ArrayRef<Operation *> slice) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(rootOp);
  FailureOr<IREE::Flow::DispatchRegionOp> dispatchRegionOp =
      IREE::Flow::wrapOpInDispatchRegion(rewriter, rootOp);
  if (failed(dispatchRegionOp)) {
    return rootOp->emitOpError("failed to form dispatch region with root op");
  }

  // todo：这个操作的核心是slice是如何计算出来的。
  FailureOr<IREE::Flow::DispatchRegionOp> newDispatchOp =
      movePrecedingOpsIntoDispatchRegion(rewriter, slice,
                                         dispatchRegionOp.value());
  if (failed(newDispatchOp)) {
    return dispatchRegionOp.value()->emitOpError(
        "failed to move slice into op");
  }
  return newDispatchOp.value();
}

// 这个pass的函数入口。
void FormScalarDispatchesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  int scalarWorkloadLimit = 1;
  // Convenient struct to hold all operations that need to be moved into a
  // descriptor.
  // 辅助数据结构，维护一个region里的所有operation。
  struct DispatchRegionDescriptor {
    Operation *rootOp;
    SmallVector<Operation *> fusedOps;
  };

  SmallVector<DispatchRegionDescriptor> dispatches;
  llvm::DenseMap<Operation *, Operation *> opToRootMap;

  // 使用逆后序遍历，是compiler遍历计算图的常用方式
  // 可以保证能够按照循环依赖的方式遍历
  // Walk the function in postorder, reverse orded ignore all operations
  // not immediately nested within the `funcOp`.
  funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (op->getParentOp() != funcOp || opToRootMap.count(op)) {
      return;
    }

    // scalar只支持int，index，double以及tensor<1>
    if (!isSliceRoot(scalarWorkloadLimit, op)) {
      return;
    }

    // 这段代码就是用computeSliceToMoveIntoDispatch这个函数
    // 来结算如何分配dispatch
    // fusedOpsSet中存储好能够跟rootOp fuse到一个dispatch region的operations
    // 这个函数主要依靠backward slice反向切片技术，来
    // 尝试找到fuse的candidates
    // backward slice技术是找寻所有rootOp的def传递链条，搭配上
    // filter的过滤限制，得到可以fuse到一个region的operation集和。
    llvm::SetVector<Operation *> fusedOpsSet =
        computeSliceToMoveIntoDispatch(scalarWorkloadLimit, op, opToRootMap);
    for (Operation *sliceOp : fusedOpsSet) {
      assert(!opToRootMap.count(sliceOp) &&
             "trying to add same op to two dispatches");
      // 添加得到的fusedOpsSet到rootOp的dispatch region中
      // 这个fuse是比较强条件的fuse
      // 要求currentOp和rootOp必须是def-use链条关系
      opToRootMap[sliceOp] = op;
    }

    // Iterate backwards within the block to get ops that dont necessarily
    // have producer -> consumer relationship but can still be fused.
    // 先前做好了producer-consumer的融合，这里尝试进行水平融合。
    Block *currBlock = op->getBlock();
    Operation *prevOp = op;
    bool didHorizontalFusion = false;
    llvm::SetVector<Operation *> ineligibleRoots;
    while (prevOp != &currBlock->front()) {
      prevOp = prevOp->getPrevNode();

      // If this operation is used by a operation we previously visited, but we
      // couldn't fuse it, stop.
      if (ineligibleRoots.contains(prevOp)) {
        break;
      }

      if (opToRootMap.count(prevOp)) {
        continue;
      }

      if (!isSliceRoot(scalarWorkloadLimit, prevOp)) {
        if (fusedOpsSet.contains(prevOp)) {
          continue;
        }
        // If this op is not being fused, any operations that defines values
        // used by this op cannot be horizontally fused
        // Insert all operations into the set that define op's operands or
        // define values used inside of op's regions
        // 一个operation是不可以融合的
        // 则该operation的region中使用到的values的def也不可以融合
        // 该operation的operands的def也不可以融合
        mlir::visitUsedValuesDefinedAbove(
            prevOp->getRegions(), [&](OpOperand *operand) {
              if (auto definingOp = operand->get().getDefiningOp()) {
                ineligibleRoots.insert(definingOp);
              }
            });

        for (Value val : prevOp->getOperands()) {
          if (auto definingOp = val.getDefiningOp()) {
            ineligibleRoots.insert(definingOp);
          }
        }
        continue;
      }

      didHorizontalFusion = true;
      fusedOpsSet.insert(prevOp);
      opToRootMap[prevOp] = op;
      llvm::SetVector<Operation *> currSlice = computeSliceToMoveIntoDispatch(
          scalarWorkloadLimit, prevOp, opToRootMap);
      for (auto sliceOp : currSlice) {
        assert(!opToRootMap.count(sliceOp) &&
               "trying to add same op to two dispatches");
        opToRootMap[sliceOp] = op;
      }
      fusedOpsSet.insert(currSlice.begin(), currSlice.end());
    }

    // 创建dispatch discriptor（一个smallvector)
    DispatchRegionDescriptor &currDispatch =
        dispatches.emplace_back(DispatchRegionDescriptor{});
    currDispatch.rootOp = op;
    currDispatch.fusedOps.assign(fusedOpsSet.begin(), fusedOpsSet.end());
    if (didHorizontalFusion) {
      mlir::computeTopologicalSorting(currDispatch.fusedOps);
    }
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Num scalar dispatches : " << dispatches.size() << "\n";
    for (auto [index, dispatch] : llvm::enumerate(dispatches)) {
      llvm::dbgs() << "//--------------------------//\n";
      llvm::dbgs() << "Dispatch : " << index << ", Root :";
      dispatch.rootOp->print(llvm::dbgs());
      llvm::dbgs() << "\nFusedOps :";
      for (auto fusedOp : dispatch.fusedOps) {
        fusedOp->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "//--------------------------//\n";
    }
  });

  // 基于前面做好的dispatch分析，做rewriter改写
  IRRewriter rewriter(context);
  for (auto &currDispatch : dispatches) {
    rewriter.setInsertionPoint(currDispatch.rootOp);
    FailureOr<IREE::Flow::DispatchRegionOp> dispatchRegionOp =
        formDispatchRegionFromSlice(rewriter, currDispatch.rootOp,
                                    currDispatch.fusedOps);
    if (failed(dispatchRegionOp)) {
      currDispatch.rootOp->emitOpError(
          "failed to form scalar dispatch region with operation as root");
      return signalPassFailure();
    }

    // 后续会有对于workgroup count的处理
    // Set the workgroup count to {1, 1, 1} since this is to be executed
    // sequentially (at leats for now)
    Region &countRegion = dispatchRegionOp->getWorkgroupCount();
    Block *countBody = rewriter.createBlock(&countRegion, countRegion.begin());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(countBody);
    auto one = rewriter.create<arith::ConstantIndexOp>(
        dispatchRegionOp.value()->getLoc(), 1);
    rewriter.create<IREE::Flow::ReturnOp>(dispatchRegionOp.value()->getLoc(),
                                          ValueRange{one, one, one});
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
