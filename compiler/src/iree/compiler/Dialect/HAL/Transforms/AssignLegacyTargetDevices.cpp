// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_ASSIGNLEGACYTARGETDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-assign-legacy-target-devices
//===----------------------------------------------------------------------===//
// 要想理解这个pass，首先要理解HAL的device和target的概念。
// 然后理解iree/compiler/Dialect/HAL/Transforms/Passes.h.inc自动生成的代码是关键。

struct AssignLegacyTargetDevicesPass
    : public IREE::HAL::impl::AssignLegacyTargetDevicesPassBase<
          AssignLegacyTargetDevicesPass> {
  using IREE::HAL::impl::AssignLegacyTargetDevicesPassBase<
      AssignLegacyTargetDevicesPass>::AssignLegacyTargetDevicesPassBase;

  // Return the dialect that must be loaded in the context before this pass.
  // 这个函数很有意思，针对不同的backend，比如vulkan，metal，spirv，都会有对应的dialect，这里会返回这些dialect。
  // 并搭配上HAL dialect。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (StringRef name : targetRegistry->getRegisteredTargetBackends()) {
      // Registers dependent dialects for the TargetBackend.
      // Mirrors the method on mlir::Pass of the same name. A TargetBackend is
      // expected to register the dialects it will create entities for (Operations,
      // Types, Attributes).
      // 实际会由用户自定义的backend实现getDependentDialects方法，该方法将特定device需要的dialect注册到registry中。
      targetRegistry->getTargetBackend(name)->getDependentDialects(registry);
    }
  }

  /// What we can get
  /// module attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>], legacy_sync}>]} {
  /// ...
  /// }
  void runOnOperation() override {
    // assign方法作用在moduleOp上，为moduleOp添加hal.device.targets属性。
    auto moduleOp = getOperation();

    // If no targets are specified we can't do anything - another pass earlier
    // in the pipeline will have had to add the targets.
    // AssignLegacyTargetDevicesPassBase(AssignLegacyTargetDevicesPassOptions options) : AssignLegacyTargetDevicesPassBase() {
    //   targetRegistry = std::move(options.targetRegistry);
    //   targetBackends = std::move(options.targetBackends);
    // }
    // targetBackends和前面getDialect用到的targetRegistry均是从options由来。
    if (targetBackends.empty()) {
      return;
    }

    // Check to see if targets are already specified and if so then no-op the
    // pass so that we don't mess with whatever the user intended.
    auto existingTargetsAttr =
        moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (existingTargetsAttr) {
      return;
    }

    // If there are any device globals declared then bail as it means the user
    // has already materialized the devices they want.
    // assign之后才会materialize，所以这里可以通过检查是否有global来判断是否已经assign过了。
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      if (isa<IREE::HAL::DeviceType>(globalOp.getGlobalType())) {
        return;
      }
    }

    llvm::SmallDenseSet<Attribute> targetAttrSet;
    SmallVector<Attribute> targetAttrs;
    // targetBackends是string集合
    // targetRegistry是reference指针集合
    // 具体看passes.cpp中如何传入options即可：
    /* 
      ListOption<std::string> legacyTargetBackends{
        *this,
        "legacy-target-backends",
        llvm::cl::desc("DEPRECATED: Target backend names."),
        llvm::cl::ZeroOrMore,
      };
      options.targetRegistry = &targetRegistry;
    */
    for (const auto &targetBackendName : targetBackends) {
      auto targetBackend = targetRegistry->getTargetBackend(targetBackendName);

      // 用户指定了target name，但是先前没有register，这里会报错。
      if (!targetBackend) {
        auto diagnostic = emitError(moduleOp.getLoc())
                          << "target backend '" << targetBackendName
                          << "' not registered; registered backends: [";
        llvm::interleaveComma(targetRegistry->getRegisteredTargetBackends(),
                              diagnostic);
        diagnostic << "]";
        return signalPassFailure();
      }
      auto targetDeviceName = targetBackend->getLegacyDefaultDeviceID();
      auto targetDevice = targetRegistry->getTargetDevice(targetDeviceName);
      if (!targetDevice) {
        auto diagnostic = emitError(moduleOp.getLoc())
                          << "target device '" << targetDeviceName
                          << "' not registered; registered devices: [";
        llvm::interleaveComma(targetRegistry->getRegisteredTargetDevices(),
                              diagnostic);
        diagnostic << "]";
        return signalPassFailure();
      }

      // backend是cuda，device是cuda-nvptx-fb

      // Ask the target backend for its default device specification attribute.
      // 可以看cuda的backend target如何实现。
      auto targetAttr = targetDevice->getDefaultDeviceTarget(
          moduleOp.getContext(), *targetRegistry.value);
      if (!targetAttr) {
        emitError(moduleOp.getLoc()) << "no default device targets available";
        return signalPassFailure();
      }
      if (!targetAttrSet.contains(targetAttr)) {
        targetAttrSet.insert(targetAttr);
        targetAttrs.push_back(targetAttr);
      }
    }

    Attribute targetsAttr;
    if (targetAttrs.size() == 1) {
      targetsAttr = targetAttrs.front();
    } else {
      targetsAttr =
          IREE::HAL::DeviceSelectAttr::get(moduleOp.getContext(), targetAttrs);
    }
    moduleOp->setAttr("hal.device.targets",
                      ArrayAttr::get(moduleOp.getContext(), targetsAttr));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
