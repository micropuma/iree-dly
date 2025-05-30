// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREECodegenAttrs.h - Codegen dialect attributes --------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_
#define IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {
/// Typedef for tile sizes to use at different levels of tiling.
using TileSizesListType = SmallVector<SmallVector<int64_t>>;
using TileSizesListTypeRef = ArrayRef<SmallVector<int64_t>>;
/// Typedef for scalable tile flags at different levels of tiling.
using ScalableTileFlagsListType = SmallVector<SmallVector<bool>>;
using ScalableTileFlagsListTypeRef = ArrayRef<SmallVector<bool>>;
} // namespace mlir::iree_compiler

// clang-format off
#include "iree/compiler/Codegen/Dialect/Codegen/IR/LoweringConfigEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h.inc"
// clang-format on

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for getting/setting iree_codegen.translation_info attribute on the
// `hal.executable.export`
//===----------------------------------------------------------------------===//

/// Returns the translation info for the `funcOp`. Returns `nullptr` on failure.
IREE::Codegen::TranslationInfoAttr
getTranslationInfo(mlir::FunctionOpInterface funcOp);

/// Returns the workgroup size specified on the `exportOp`.
// 在translation info中指定：workgroup_size = [64, 2, 1]
std::optional<SmallVector<int64_t>>
getWorkgroupSize(mlir::FunctionOpInterface funcOp);

/// Returns the subgroup size specified on the `exportOp`.
std::optional<int64_t> getSubgroupSize(mlir::FunctionOpInterface funcOp);

/// Sets and overwites the translate executable info for the given entry point.
/// Returns failure if the given entry point is not exported via
/// hal.executable.export.
LogicalResult
setTranslationInfo(mlir::FunctionOpInterface entryPoint,
                   IREE::Codegen::TranslationInfoAttr translationInfo);

/// Erases any translation info set on an operation.
void eraseTranslationInfo(mlir::FunctionOpInterface funcOp);

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.lowering_config` attribute on root
// operations.
//===----------------------------------------------------------------------===//

static const char kConfigAttrName[] = "lowering_config";

/// Returns the lowering configuration set for an operation. Returns `nullptr`
/// if no value is set.  It expects that the attribute is stored using the
/// identifier `lowering_config`.
template <typename ConfigTy = IREE::Codegen::LoweringConfigAttrInterface>
ConfigTy getLoweringConfig(Operation *op) {
  return op->getAttrOfType<ConfigTy>(kConfigAttrName);
}

/// Returns the op that carries the `lowering_config` attribute; returns nullptr
/// if none is carrying the attribute.
///
/// This scans ops in top-down order and the first one carrying the attribute
/// will be returned.
template <typename ConfigTy = IREE::Codegen::LoweringConfigAttrInterface>
FailureOr<Operation *>
getLoweringConfigCarryingOp(ArrayRef<Operation *> computeOps) {
  for (Operation *op : computeOps) {
    if (getLoweringConfig<ConfigTy>(op))
      return op;
  }
  return failure();
}

/// Returns the lowering configuration from the list of operations; returns
/// failure if no operations is carrying a lowering config.
///
/// This scans ops in top-down order and the first one carrying the attribute
/// will be returned.
template <typename ConfigTy = IREE::Codegen::LoweringConfigAttrInterface>
FailureOr<ConfigTy> getFirstLoweringConfig(ArrayRef<Operation *> computeOps) {
  FailureOr<Operation *> op = getLoweringConfigCarryingOp<ConfigTy>(computeOps);
  if (failed(op))
    return failure();
  return getLoweringConfig<ConfigTy>(*op);
}

/// Returns the tile sizes for a particular operation if the
/// `iree_codegen.lowering_config` attribute is set on it.
SmallVector<int64_t> getTileSizes(Operation *op, unsigned level);
SmallVector<Value> getTileSizes(OpBuilder &b, Operation *op, unsigned level);

/// Sets the lowering configuration, overwriting existing attribute values.
void setLoweringConfig(Operation *op, Attribute config);

/// Convenience function that sets the lowering configuration on the operation
/// and translation info.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    mlir::FunctionOpInterface entryPointFn, Operation *op,
    IREE::Codegen::LoweringConfigAttrInterface config,
    IREE::Codegen::TranslationInfoAttr translationInfo) {
  if (config) {
    setLoweringConfig(op, config);
  }
  if (translationInfo) {
    (void)setTranslationInfo(entryPointFn, translationInfo);
  }
  return success();
}

/// Convenience function that sets the lowering configuration on the operation
/// and translation info for a generic lowering config, lowering pipeline,
/// and optional workgroup/subgroup size.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    mlir::FunctionOpInterface entryPointFn, Operation *op,
    IREE::Codegen::LoweringConfigAttrInterface config,
    IREE::Codegen::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize = {},
    std::optional<int64_t> subgroupSize = {},
    DictionaryAttr pipelineConfig = DictionaryAttr()) {
  MLIRContext *context = entryPointFn.getContext();
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context, passPipeline, SymbolRefAttr(), workgroupSize, subgroupSize,
      pipelineConfig);
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, config,
                                               translationInfo);
}

/// Convenience function that sets the lowering configuration on the operation
/// and translation info on the entry point op for the common case of specifying
/// tile sizes to use for the operation, and pass pipeline to use for the
/// translation.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    mlir::FunctionOpInterface entryPointFn, Operation *op,
    TileSizesListTypeRef tileSizes,
    ScalableTileFlagsListTypeRef scalableTileFlags,
    IREE::Codegen::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize = {},
    std::optional<int64_t> subgroupSize = {},
    DictionaryAttr pipelineConfig = DictionaryAttr()) {
  MLIRContext *context = entryPointFn.getContext();
  auto config = IREE::Codegen::LoweringConfigAttr::get(context, tileSizes,
                                                       scalableTileFlags);
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, config,
                                               passPipeline, workgroupSize,
                                               subgroupSize, pipelineConfig);
}

/// Overload of setOpConfigAndEntryPointFnTranslation() for the "no scalable
/// flags" case.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    mlir::FunctionOpInterface entryPointFn, Operation *op,
    TileSizesListTypeRef tileSizes,
    IREE::Codegen::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize = {},
    std::optional<int64_t> subgroupSize = {},
    DictionaryAttr pipelineConfig = DictionaryAttr()) {
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, tileSizes, {},
                                               passPipeline, workgroupSize,
                                               subgroupSize, pipelineConfig);
}

/// Function to erase lowering configs that are set on an operation.
void eraseLoweringConfig(Operation *op);

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.compilation_info` attribute on root
// operations to override IREEs default compilation.
//===----------------------------------------------------------------------===//

/// Returns the `#iree_codegen.compilation_info` set on the operation. Assumes
/// that the identifier used is `compilation_info`.
IREE::Codegen::CompilationInfoAttr getCompilationInfo(Operation *op);

/// Sets the `config` to use for compiling the operation. If `op` is the root
/// operation of the dispatch region, overrides the default configuration that
/// is used for compilation.
void setCompilationInfo(Operation *op,
                        IREE::Codegen::CompilationInfoAttr config);

/// Removes the `#iree_codegen.compilation_info` attribute that is set on the
/// operation.
void eraseCompilationInfo(Operation *op);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_
