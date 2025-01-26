// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE                                                             \
  "iree-dispatch-creation-convert-dispatch-regions-to-workgroups"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CONVERTDISPATCHREGIONSTOWORKGROUPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct ConvertDispatchRegionsToWorkgroupsPass
    : public impl::ConvertDispatchRegionsToWorkgroupsPassBase<
          ConvertDispatchRegionsToWorkgroupsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void ConvertDispatchRegionsToWorkgroupsPass::runOnOperation() {
  // mlir编写的方法：
  // 获取想要runOn的operation类型
  // 定义rewriter，并传入operation类型
  FunctionOpInterface funcOp = getOperation();
  /// Simplfy the given tensor::DimOps as much as possible.
  /// * Static dimensions are replaced by constant.
  /// * Dynamic dim ops are pushed as much as possible to the top of the function,
  ///   i.e., if the dim of a value is known to be equal to the dim of a value on
  ///   the reverse SSA use-def chain, rewrite the value with a dim op of that
  ///   value.
  TensorDimTrackingRewriter rewriter(funcOp);

  // 用一个SmallVector存储funOp里面出现的所有Flow::DispatchRegionOp
  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  funcOp.walk(
      [&](IREE::Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  // 统计总共有多少regionOps
  numDispatches += regionOps.size();

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  for (auto regionOp : regionOps) {
    // dispatch to workgroup的核心算法
    auto maybeWorkgroupOp =
        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) {
      regionOp.emitError(
          "failed to convert dispatch.region op to dispatch.workgroup op");
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
