// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-spirv-map-memref-storage-class"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVMAPMEMREFSTORAGECLASSPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

template <bool UseIndirectBindings>
std::optional<spirv::StorageClass>
mapHALDescriptorTypeForVulkan(Attribute attr) {
  if (auto dtAttr =
          llvm::dyn_cast_if_present<IREE::HAL::DescriptorTypeAttr>(attr)) {
    switch (dtAttr.getValue()) {
    case IREE::HAL::DescriptorType::UniformBuffer:
      return spirv::StorageClass::Uniform;
    case IREE::HAL::DescriptorType::StorageBuffer:
      return UseIndirectBindings ? spirv::StorageClass::PhysicalStorageBuffer
                                 : spirv::StorageClass::StorageBuffer;
    default:
      return std::nullopt;
    }
  }
  if (auto gpuAttr = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(attr)) {
    switch (gpuAttr.getValue()) {
    case gpu::AddressSpace::Workgroup:
      return spirv::StorageClass::Workgroup;
    default:
      return std::nullopt;
    }
  };
  return spirv::mapMemorySpaceToVulkanStorageClass(attr);
}

std::optional<spirv::StorageClass>
mapHALDescriptorTypeForOpenCL(Attribute attr) {
  if (auto dtAttr =
          llvm::dyn_cast_if_present<IREE::HAL::DescriptorTypeAttr>(attr)) {
    switch (dtAttr.getValue()) {
    case IREE::HAL::DescriptorType::UniformBuffer:
      return spirv::StorageClass::Uniform;
    case IREE::HAL::DescriptorType::StorageBuffer:
      return spirv::StorageClass::CrossWorkgroup;
    }
  }
  if (auto gpuAttr = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(attr)) {
    switch (gpuAttr.getValue()) {
    case gpu::AddressSpace::Workgroup:
      return spirv::StorageClass::Workgroup;
    default:
      return std::nullopt;
    }
  };
  return spirv::mapMemorySpaceToOpenCLStorageClass(attr);
}

bool allowsShaderCapability(ArrayRef<StringRef> features) {
  for (StringRef feature : features) {
    if (feature.consume_front("cap:") && feature == "Shader")
      return true;
  }
  return false;
}

bool allowsKernelCapability(ArrayRef<StringRef> features) {
  for (StringRef feature : features) {
    if (feature.consume_front("cap:") && feature == "Kernel")
      return true;
  }
  return false;
}

struct SPIRVMapMemRefStorageClassPass final
    : impl::SPIRVMapMemRefStorageClassPassBase<SPIRVMapMemRefStorageClassPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    bool useIndirectBindings = usesIndirectBindingsAttr(op);

    spirv::MemorySpaceToStorageClassMap memorySpaceMap;

    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(op)) {
      SmallVector<StringRef> features;
      llvm::SplitString(target.getFeatures(), features, ",");
      if (allowsShaderCapability(features)) {
        memorySpaceMap = useIndirectBindings
                             ? &mapHALDescriptorTypeForVulkan<true>
                             : &mapHALDescriptorTypeForVulkan<false>;
      } else if (allowsKernelCapability(features)) {
        memorySpaceMap = mapHALDescriptorTypeForOpenCL;
      }
    }
    if (!memorySpaceMap) {
      op->emitError("missing storage class map for unknown client API");
      return signalPassFailure();
    }

    spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);
    // Perform the replacement.
    spirv::convertMemRefTypesAndAttrs(op, converter);

    // Check if there are any illegal ops remaining.
    std::unique_ptr<ConversionTarget> target =
        spirv::getMemorySpaceToStorageClassTarget(*context);
    op->walk([&target, this](Operation *childOp) {
      if (target->isIllegal(childOp)) {
        childOp->emitOpError("failed to legalize memory space");
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
