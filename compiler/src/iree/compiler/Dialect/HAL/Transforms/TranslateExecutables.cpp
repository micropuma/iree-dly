// Copyright 2020 The IREE Authors
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
#include "iree/compiler/Utils/TracingUtils.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_TRANSLATEEXECUTABLESPASS
#define GEN_PASS_DEF_TRANSLATETARGETEXECUTABLEVARIANTSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-translate-target-executable-variants
//===----------------------------------------------------------------------===//

struct TranslateTargetExecutableVariantsPass
    : public IREE::HAL::impl::TranslateTargetExecutableVariantsPassBase<
          TranslateTargetExecutableVariantsPass> {
  using IREE::HAL::impl::TranslateTargetExecutableVariantsPassBase<
      TranslateTargetExecutableVariantsPass>::
      TranslateTargetExecutableVariantsPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto variantOp = getOperation();
    if (variantOp.getTarget().getBackend().getValue() != target)
      return;
    if (variantOp.isExternal())
      return;

    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (!targetBackend) {
      variantOp.emitError() << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    OpPassManager passManager(variantOp.getOperationName());

    // 这里做dipatch，dispatch成特定backend实现的translate pass
    // 以cuda为例，这里就会是cuda生成的pass
    targetBackend->buildTranslationPassPipeline(variantOp.getTargetAttr(),
                                                passManager);
    if (failed(runPipeline(passManager, variantOp))) {
      variantOp.emitError() << "failed to run translation of source "
                               "executable to target executable for backend "
                            << variantOp.getTarget();
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-hal-translate-executables
//===----------------------------------------------------------------------===//

struct TranslateExecutablesPass
    : public IREE::HAL::impl::TranslateExecutablesPassBase<
          TranslateExecutablesPass> {
  using IREE::HAL::impl::TranslateExecutablesPassBase<
      TranslateExecutablesPass>::TranslateExecutablesPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // 只用注册HAL和bufferization两个dialect
    // 以及targetRegistry中的所有target backend的dependent dialects
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    auto targetBackends = targetRegistry->getTargetBackends(
        targetRegistry->getRegisteredTargetBackends());
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(executableOp)) {
      passManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
        // Translates an executable variant for a specific target from its generic
        // MLIR dialects (such as `linalg`) to the target-specific dialects (`llvm`,
        // spirv`, etc).
          IREE::HAL::createTranslateTargetExecutableVariantsPass(
              {targetRegistry, targetName}));
    }

    IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(INFO, executableOp.getSymName().str());

    if (failed(runPipeline(passManager, executableOp))) {
      llvm::errs() << "failed to translate executables\n";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
