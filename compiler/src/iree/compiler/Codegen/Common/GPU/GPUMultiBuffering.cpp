// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUMULTIBUFFERINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

// todo hal: 一个比较重要，且简单的pass
// 这个Pass是完成软件流水线的重要一环。
// 多缓冲模式中应使用的缓冲数量取决于流水线深度（也称流水线阶段数量）
// 其核心思想是通过引入多缓冲，使得循环迭代尽可能并行执行，从而为最终实现软件流水线创造条件。
namespace {
struct GPUMultiBufferingPass final
    : impl::GPUMultiBufferingPassBase<GPUMultiBufferingPass> {
  using GPUMultiBufferingPassBase::GPUMultiBufferingPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // First hoist all shared memory allocations to the entry block of the
    // function. We can see memref.alloc in loops after bufferizing scf.forall
    // with promoted shared memory usage inside.

    // 收集所有allocOp，并且是shared memory中开辟的
    // 提前到func头部
    SmallVector<memref::AllocOp> allocs;
    // Collect all the alloc operations.
    // 收集所有在shared memory中的内存空间开辟
    funcOp.walk([&](memref::AllocOp allocOp) {
      // 如果是shared memory的分配，那么就将其移动到函数的entry block中
      // 即multi-buffering优化是只针对shared memory的分配
      if (hasSharedMemoryAddressSpace(allocOp.getType()))
        allocs.push_back(allocOp);
    });

    assert(funcOp.getBlocks().size() == 1);
    // 将所有allocOp放在funcOp的最前面
    for (memref::AllocOp allocOp : allocs) {
      if (allocOp->getParentOp() != funcOp)
        allocOp->moveBefore(&*funcOp.begin()->begin());
    }

    // Then perform multibuffering transformations.

    allocs.clear();
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      // Skip allocations not used in a loop.
      // 学习这里的写法：获取user list，以及判断user的所属operation
      for (Operation *user : allocOp->getUsers()) {
        auto loop = user->getParentOfType<scf::ForOp>();
        if (!loop)
          return WalkResult::advance();
      }
      allocs.push_back(allocOp);
    
      // Interrupt: the walk will be interrupted and no more operations, regions or blocks will be visited.
      // Advance: the walk will continue.
      // Skip: the walk of the current operation, region or block and their nested elements that haven't been visited already will be skipped and will continue with the next operation, region or block.
      return WalkResult::advance();
    });
    // Apply multi-buffering to all of them.
    for (memref::AllocOp alloc : allocs) {
      // 这个pass的真正核心步骤，完成多缓冲的引入
      // 辅助gpu软件流水线的设计
      // 其核心是：
      // 1. alloc op的使用者必须在for loop里
      // 2. alloc op的使用者不能有loop-carried dependence
      
      if (failed(memref::multiBuffer(alloc, numBuffers))) {
        // Error out and stop if any buffer cannot be multi buffered, as future
        // software pipelining transformations will assume this happened.
        alloc.emitOpError("cannot be multi-buffered");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
