// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREDUCEBANKCONFLICTSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Check if AllocOp has a CollapseShapeOp user.
static bool hasCollapseShapeUser(memref::AllocOp allocOp) {
  SmallVector<Operation *> users(allocOp->getUsers());

  // 对于一个allocOp操作的每个user都要做检查，包括subview user传播user链条
  while (!users.empty()) {
    auto user = users.pop_back_val();
    if (isa<memref::CollapseShapeOp>(user)) {
      return true;
    }
    if (isa<ViewLikeOpInterface>(user)) {
      for (auto u : user->getUsers()) {
        users.push_back(u);
      }
    }
  }
  return false;
}

/// Pad out the inner dimension of the `memref.alloc` op in order reduce the
/// chances to have bank conflicts when reading 2D shapes within shared memory.
/// 注意，内存的padding是在memory的最内层维度做
static void padAlloc(MLIRContext *context, memref::AllocOp allocOp,
                     unsigned paddingSizeBits) {
  auto allocOpShape = allocOp.getType().getShape();
  if (allocOpShape.empty())
    return;
  int64_t innerDim = allocOpShape.back();
  if (ShapedType::isDynamic(innerDim))
    return;

  // Return if we have CollapseShape op as an user as padding in that case is
  // unsupported.
  // CollapseShapeOp会压缩某个维度，我们的padding暂时不支持对alloc做padding，然后再压缩
  if (hasCollapseShapeUser(allocOp))
    return;

  // 获取allocOp的数值类型
  Type elType = allocOp.getType().getElementType();
  unsigned bitwidth =
      mlir::DataLayout::closest(allocOp).getTypeSizeInBits(elType);
  // Pad with the specified amount. This should be >= bank size and <= widest
  // load size.
  int64_t paddingSize = paddingSizeBits / bitwidth;
  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  shape.back() = shape.back() + paddingSize;
  MemRefType allocType =
      MemRefType::get(shape, elType, MemRefLayoutAttrInterface{},
                      allocOp.getType().getMemorySpace());
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(allocOp);
  Location loc = allocOp.getLoc();
  Value paddedAlloc = rewriter.create<memref::AllocOp>(loc, allocType);
  SmallVector<int64_t> offsets(shape.size(), 0);
  SmallVector<int64_t> strides(shape.size(), 1);
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, paddedAlloc, offsets, allocOp.getType().getShape(), strides);
  
  // 替换存储的使用，并传播类型
  replaceMemrefUsesAndPropagateType(rewriter, loc, allocOp, subview);
  rewriter.eraseOp(allocOp);
}

/// Pass to reduce the number of bank conflicts when accessing shared memory in
/// a 2D manner. This is a simple version just padding allocation.
/// This doesn't fully remove bank conflicts and increase the shared memory
/// usage. In order to get better memory access patterns we should do shared
/// memory swizzling which requires more complex transformations. This pass can
/// be removed once the better solution is implemented.
struct GPUReduceBankConflictsPass final
    : impl::GPUReduceBankConflictsPassBase<GPUReduceBankConflictsPass> {
  using GPUReduceBankConflictsPassBase::GPUReduceBankConflictsPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (failed(reduceSharedMemoryBankConflicts(funcOp, paddingBits)))
      signalPassFailure();
  }
};

} // namespace

/// 核心padding逻辑
LogicalResult reduceSharedMemoryBankConflicts(mlir::FunctionOpInterface funcOp,
                                              unsigned paddingSize) {
  SmallVector<memref::AllocOp> sharedMemAllocs;
  // Collect all the alloc operations.
  // 注意，只有在共享内存中的才收集
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (hasSharedMemoryAddressSpace(allocOp.getType()) &&
        allocOp.getType().hasStaticShape()) {
      sharedMemAllocs.push_back(allocOp);
    }
  });
  for (memref::AllocOp alloc : sharedMemAllocs)
    // 对于共享内存中的alloc操作，进行padding
    padAlloc(funcOp->getContext(), alloc, paddingSize);

  // In the current form this always succeeds.
  return success();
}

} // namespace mlir::iree_compiler
