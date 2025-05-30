// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_COMMON_GPU_PASSES
#define IREE_CODEGEN_COMMON_GPU_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// Common Passes used for GPU-like backends (keep alphabetical)
//===---------------------------------------------------------------------===//

def GPUCheckResourceUsagePass :
    InterfacePass<"iree-codegen-gpu-check-resource-usage", "mlir::FunctionOpInterface"> {
  let summary = "Checks GPU specific resource usage constraints like shared memory limits";
  let constructor = "mlir::iree_compiler::createGPUCheckResourceUsagePass()";
}

def GPUCombineValueBarriersPass :
    Pass<"iree-codegen-gpu-combine-value-barriers", ""> {
  let summary = "Combines `iree_gpu.value_barrier` ops";
  let dependentDialects = ["::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"];
}

def GPUCreateFastSlowPathPass :
    InterfacePass<"iree-codegen-gpu-create-fast-slow-path", "mlir::FunctionOpInterface"> {
  let summary = "Create separate fast and slow paths to handle padding";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def GPUDistributeForallPass :
    InterfacePass<"iree-codegen-gpu-distribute-forall", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute scf.forall ops.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUDistributePass :
    InterfacePass<"iree-codegen-gpu-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute scf.forall ops using upstream patterns.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUDistributeSharedMemoryCopyPass :
    InterfacePass<"iree-codegen-gpu-distribute-shared-memory-copy", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute shared memory copies to threads.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect", "::mlir::vector::VectorDialect"
  ];
}

def GPUDistributeScfForPass :
    InterfacePass<"iree-codegen-gpu-distribute-scf-for", "mlir::FunctionOpInterface"> {
  let summary = "Distribute tiled loop nests to invocations";
  let dependentDialects = ["::mlir::gpu::GPUDialect"];
  let options = [
    Option<"useBlockDims", "use-block-dims", "bool",
           /*default=*/"true",
           "Use gpu.block_dim ops to query distribution sizes.">,
  ];
}

def GPUGeneralizeNamedOpsPass :
    InterfacePass<"iree-codegen-gpu-generalize-named-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert named Linalg ops to linalg.generic ops";
}

def GPUGreedilyDistributeToThreadsPass :
    InterfacePass<"iree-codegen-gpu-greedily-distribute-to-threads", "mlir::FunctionOpInterface"> {
  let summary = "Greedily distributes all remaining tilable ops to threads";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUInferMemorySpacePass :
    InterfacePass<"iree-codegen-gpu-infer-memory-space", "mlir::FunctionOpInterface"> {
  let summary = "Pass to infer and set the memory space for all alloc_tensor ops.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect"
  ];
}

def GPULowerToUKernelsPass :
    Pass<"iree-codegen-gpu-lower-to-ukernels", ""> {
  let summary = "Separate out parts of the IR that lower to a micro-kernel";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUMultiBufferingPass :
    InterfacePass<"iree-codegen-gpu-multi-buffering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to do multi buffering.";
  let dependentDialects = ["::mlir::affine::AffineDialect"];
  let options = [
    Option<"numBuffers", "num-buffers", "unsigned",
            /*default=*/"5",
            "Number of buffers to use.">,
  ];
}

def GPUPipeliningPass :
    InterfacePass<"iree-codegen-gpu-pipelining", "mlir::FunctionOpInterface"> {
  let summary = "Pass to do software pipelining.";
  let options = [
    Option<"epiloguePeeling", "epilogue-peeling", "bool",
            /*default=*/"true",
           "Try to use un-peeling epilogue when false, peeled epilouge o.w.">,
    Option<"depth", "pipeline-depth", "int64_t",
            /*default=*/"2",
           "Number of stages ">,
    Option<"scheduleIndex", "schedule-index", "int64_t",
            /*default=*/"0",
           "Allows picking different schedule for the pipelining transformation.">,
    Option<"transformFileName", "transform-file-name", "std::string",
            /*default=*/"\"\"",
            "Optional filename containing a transform dialect specification to "
            "apply. If left empty, the IR is assumed to contain one top-level "
            "transform dialect operation somewhere in the module.">,
  ];
}

def GPUPromoteMatmulOperandsPass :
    InterfacePass<"iree-codegen-gpu-promote-matmul-operands",
                  "mlir::FunctionOpInterface"> {
  let summary = "Pass to insert copies with a different thread configuration "
                "on matmul operands";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def GPUReduceBankConflictsPass :
    InterfacePass<"iree-codegen-gpu-reduce-bank-conflicts", "mlir::FunctionOpInterface"> {
  let summary = "Pass to try to reduce the number of bank conflicts by padding memref.alloc ops.";
  let options = [
    Option<"paddingBits", "padding-bits", "unsigned",
            /*default=*/"128",
            "Padding size (in bits) to introduce between rows.">
  ];
}

def GPUReuseSharedMemoryAllocsPass :
    InterfacePass<"iree-codegen-gpu-reuse-shared-memory-allocs", "mlir::FunctionOpInterface"> {
  let summary = "Pass to reuse shared memory allocations with no overlapping liveness.";
  let dependentDialects = [
    "::mlir::nvgpu::NVGPUDialect"
  ];
}

def GPUTensorAllocPass :
    InterfacePass<"iree-codegen-gpu-tensor-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to create allocations for some tensor values to use"
                "GPU shared memory";
  let constructor = "mlir::iree_compiler::createGPUTensorAlloc()";
  let dependentDialects = ["::mlir::bufferization::BufferizationDialect"];
}

def GPUTensorTilePass :
    InterfacePass<"iree-codegen-gpu-tensor-tile", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile tensor (linalg) ops within a GPU workgroup";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"distributeToSubgroup", "distribute-to-subgroup", "bool",
           /*default=*/"false",
           "Distribute the workloads to subgroup if true, otherwise distribute to threads.">,
  ];
}

def GPUApplyTilingLevelPass :
    InterfacePass<"iree-codegen-gpu-apply-tiling-level", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile tensor ops based on tiling configs";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect", "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"tilingLevel", "tiling-level", "IREE::GPU::TilingLevel",
           /*default=*/"IREE::GPU::TilingLevel::Reduction",
           "Tiling level to tile. Supported levels are 'reduction' and 'thread'",
           [{llvm::cl::values(
              clEnumValN(IREE::GPU::TilingLevel::Reduction, "reduction",
                         "Tile and fuse all annotated ops to serial loops"),
              clEnumValN(IREE::GPU::TilingLevel::Thread, "thread",
                         "Tile and fuse all annotated ops to threads"),
              clEnumValN(IREE::GPU::TilingLevel::Subgroup, "subgroup",
                         "Tile and fuse all annotated ops to threads")
           )}]>,
    Option<"allowZeroSlices", "allow-zero-slices", "bool",
           /*default=*/"false",
           "Allow pad fusion to generate zero size slices">
  ];
}

def GPUMaterializeHostEncodingPass :
    Pass<"iree-codegen-gpu-materialize-host-encoding", "mlir::ModuleOp"> {
  let summary = "Materialize the encoding for tensor as specified by the backend.";
}

def GPUMaterializeDeviceEncodingPass :
    InterfacePass<"iree-codegen-gpu-materialize-device-encoding", "mlir::FunctionOpInterface"> {
  let summary = "Materialize the encoding for tensor as specified by the backend.";
}

def GPUTensorTileToSerialLoopsPass :
    InterfacePass<"iree-codegen-gpu-tensor-tile-to-serial-loops", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile reduction dimensions for certain GPU ops";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
  let options = [
    Option<"coalesceLoops", "coalesce-loops", "bool", /*default=*/"false",
           "Collapse the loops that are generated to a single loops">,
  ];
}

def GPUTilePass : InterfacePass<"iree-codegen-gpu-tile", "mlir::FunctionOpInterface"> {
  let summary = "Tile Linalg ops with tensor semantics to invocations";
}

def GPUTileReductionPass :
    InterfacePass<"iree-codegen-gpu-tile-reduction", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile linalg reduction dimensions.";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def GPUVerifyDistributionPass :
    InterfacePass<"iree-codegen-gpu-verify-distribution", "mlir::FunctionOpInterface"> {
  let summary = "Pass to verify writes before resolving distributed contexts.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
  ];
}

def GPUVectorAllocPass :
    InterfacePass<"iree-codegen-gpu-vector-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to create allocations for contraction inputs to copy "
                "to GPU shared memory";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def ReorderWorkgroupsPass :
    InterfacePass<"iree-codegen-reorder-workgroups", "mlir::FunctionOpInterface"> {
  let summary = "Reorder workgroup ids for better cache reuse";
  let constructor = "mlir::iree_compiler::createReorderWorkgroups()";
  let dependentDialects = ["::mlir::affine::AffineDialect"];
  let options = [
    Option<"strategy", "strategy", "std::string", /*default=*/"",
           "Workgroup reordering strategy, one of: '' (none),  'transpose'">,
  ];
}

def VectorReductionToGPUPass :
    InterfacePass<"iree-codegen-vector-reduction-to-gpu", "mlir::FunctionOpInterface"> {
  let summary = "Convert vector reduction to GPU ops.";
  let constructor = "mlir::iree_compiler::createConvertVectorReductionToGPUPass()";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect",
  ];
}

def ExpandGPUOpsPass :
    InterfacePass<"iree-codegen-expand-gpu-ops", "mlir::FunctionOpInterface"> {
  let summary = "Expands high-level GPU ops, such as clustered gpu.subgroup_reduce.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect"
  ];
}

#endif // IREE_CODEGEN_COMMON_GPU_PASSES
