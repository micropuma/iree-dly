// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

/*
  主要用于验证在 GPU 上针对矩阵乘法（matmul）和批量矩阵乘法（batchmatmul）的编译配置是否合理，
  特别是针对 CUDA 和 Tensor Core 管线的配置。
  1. workloadsize要能整除workgroup
  2. workgroup要能整除warp
  3. 一个warp的shape要能整除指令
*/

namespace mlir::iree_compiler {

// 十分有趣的code部分，重点是对于iree finetune部分的配置的合理性verify：
// #compilation0 = #iree_codegen.compilation_info<
// lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
// translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
// , { pipeline_depth = 3, store_stage = 1}>>
using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

////////////////////////////////////////////////////////////////////////////////
// Constants used in the matmul lowering verifiers.
// verify的层级，有workgroup层级，thread层级，warp层级，指令层级。
// 目前target workgroup层级
constexpr unsigned kWorkgroupTileLevel = 0;

// Use the constexpr to convey the meaning of the indices.
// Dimenstions identifiers for: workgroup size (x, y, z), and thread (x, y, z).
// 对于 workgroup 或线程维度，分别为 x, y, z 维度：
constexpr int kDimX = 0;
constexpr int kDimY = 1;
constexpr int kDimZ = 2;

// Dimensions identifiers for: matmul problem shapes (m, n, k), thread block
// shape (m, n, k), warp shape, and instruction shape (m, n, k).
// 对于矩阵乘法的问题尺寸和 tile 尺寸，分别代表矩阵的行 (M)、列 (N) 以及累加的内积维度 (K)：
constexpr int kM = 0;
constexpr int kN = 1;
constexpr int kK = 2;
////////////////////////////////////////////////////////////////////////////////

/// Returns the shape of the math instruction for the given pipeline and input
/// element type.
/// 例如，对于 Tensor Core 管线，f16 和 bf16 类型对应的指令形状为 {16, 16, 16}，而 f32 类型对应 {16, 16, 8}。
/// Tensor Core的计算能力是硬件定死的，因此我们需要根据输入的数据类型来选择合适的指令形状。
static LogicalResult
getInstructionShape(Operation *op, CodeGenPipeline pipeline,
                    Type inputElementType,
                    SmallVector<int64_t> &instructionShape) {
  switch (pipeline) {
  case CodeGenPipeline::LLVMGPUMatmulSimt:
    // SIMT Pipeline / CUDA Cores
    instructionShape = {1, 1, 1};
    break;

  // 我们的pipeline是LLVMGPUMatmulTensorCore
  case CodeGenPipeline::LLVMGPUMatmulTensorCore:
    // Tensor Core Pipeline / WMMA API
    // 对于F16和BF16，tensorcore是16x16x16
    if (inputElementType.isF16() || inputElementType.isBF16()) {
      instructionShape = {16, 16, 16};
    // 对于F32，tensorcore是16x16x8
    } else if (inputElementType.isF32()) {
      instructionShape = {16, 16, 8};
    } else {
      return op->emitError(
          "Expected f16, bf16 or f32 for Tensor Core (WMMA) pipeline");
    }
    break;
  case CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync:
    // Tensor Core Pipeline / MMA.SYNC
    if (inputElementType.isF16() || inputElementType.isBF16()) {
      instructionShape = {16, 8, 16};
    } else if (inputElementType.isF32()) {
      instructionShape = {16, 8, 8};
    } else {
      return op->emitError(
          "Expected f16, bf16 or f32 for Tensor Core (MMA.SYNC) pipeline");
    }
    break;
  default:
    return op->emitError(
        "Expected matmul SIMT, TensorCore(WMMA), or TensorCore(MMA.SYNC), "
        "compilation pipeline");
  }
  return success();
}

/// Verifies launch configuration for matmul and batchmatmul on a GPU for CUDA
/// and Tensor Core pipelines.
/*
函数 verifyGPUMatmulPipeline 用于验证 GPU 上针对 matmul 或 batch matmul 操作的编译配置是否正确。主要验证内容包括：
 1. 工作组（workgroup）大小是否设置合理；
 2. 软件流水线（software pipeline）参数是否正确配置（例如 pipeline depth 和 store stage）；
 3. 矩阵乘法的维度与 tile（分块）尺寸是否匹配；
 4. 对于 Tensor Core 管线，还要验证线程块内 warp 数量以及指令形状和 warp tile 是否能够整除。
*/
LogicalResult
verifyGPUMatmulPipeline(Operation *op,
                        IREE::Codegen::LoweringConfigAttr loweringConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo,
                        ArrayRef<int64_t> workgroupSize) {
  // This verifier only applies to matmul.
  CodeGenPipeline pipeline = translationInfo.getDispatchLoweringPassPipeline();

  // CUDA Core模式
  // Tensor Core模式
  // Tensor Core + MMA.SYNC模式
  if (pipeline != CodeGenPipeline::LLVMGPUMatmulSimt &&
      pipeline != CodeGenPipeline::LLVMGPUMatmulTensorCore &&
      pipeline != CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync) {
    return success();
  }
  // Only verify batched and unbatched matmul.
  // 只支持对于批矩阵乘和矩阵乘运算做codegen matmul
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }

  // Early exit if the workgroup size is not set.
  // 需要设定好workgroup size
  if (workgroupSize.empty()) {
    return op->emitOpError("expected workgroup size for GPU pipelines");
  }

  FailureOr<int64_t> maybeDepth =
      getSoftwarePipelineDepth(translationInfo.getConfiguration());
  FailureOr<int64_t> maybeStage =
      getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
  if (failed(maybeDepth) || failed(maybeStage)) {
    return op->emitOpError(
        "invalid matmul configuration without pipelining config");
  }

  // iree默认写入workgroup的阶段是pipeline1。
  if (*maybeStage != 1) {
    return op->emitError(
        "store to workgroup memory currently expected to happen in stage 1 of "
        "software pipeline.");
  }

  // Get compilation pipeline.
  StringRef pipelineName = stringifyEnum(pipeline);

  // Get Operand/Result types.
  // 矩阵乘运算的左右type必须保持一致，目前不支持混合精度
  mlir::Type lhsType = op->getOperand(0).getType();
  mlir::Type rhsType = op->getOperand(1).getType();
  assert(cast<ShapedType>(lhsType).getElementType() ==
             cast<ShapedType>(rhsType).getElementType() &&
         "expected lhs and rhs to have same type. Mixed input types are not "
         "supported yet in IREE Codegen.");

  // Get lhs and rhs shapes.
  // 十分经典的shape强转换代码。
  ArrayRef<int64_t> lhsShape = llvm::cast<ShapedType>(lhsType).getShape();
  ArrayRef<int64_t> rhsShape = llvm::cast<ShapedType>(rhsType).getShape();

  // Tile shapes in number of elements.
  // 获取tile配置，默认为thread block形状。
  SmallVector<int64_t> tileShape =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> threadBlockShape{tileShape};

  if (auto batchMatmulOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
    // Inspect the batch tile dimensions separately for batch. The batch tile
    // dim should be strictly greater than 1 for parallelizable loops and 0
    // for non-parallelizable.
    // 对于批量矩阵乘法，我们需要单独检查批量维度。对于可并行化的循环，批量维度应该严格大于 1，
    // 对于不可并行化的循环，批量维度应该为 0。
    if (cast<PartitionableLoopsInterface>(op).getPartitionableLoops(
            kNumMaxParallelDims)[0] == 0) {
      if (tileShape[0] > 1) {
        return op->emitError("Received batch tile dimension of ")
               << tileShape[0]
               << " instead of 1 or lower for partitionable loops with "
               << "compilation pipeline " << pipelineName;
      }
    } else {
      // 如果不可并行化，则批量维度应该为 0。
      if (tileShape[0] != 0) {
        return op->emitError("Received batch tile dimension of ")
               << tileShape[0]
               << " instead of 0 for non-partitionable loops with compilation"
               << " pipeline " << pipelineName;
      }
    }

    // Remove the batch dimension from the threadBlockShape, lhsShape, and
    // rhsShape.
    threadBlockShape = {tileShape[1], tileShape[2], tileShape[3]};

    // 选择丢弃第一个维度，即批量维度。
    lhsShape = lhsShape.drop_front();
    rhsShape = rhsShape.drop_front();
  }

  //
  // Begin verification for CUDA and Tensor Core pipelines.
  //

  // Verify the total number of threads in a thread block.
  // 计算总线程数，不能超过1024.
  // 一个thread block的threads数是1024.
  int totalNumThreads = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];

  if (totalNumThreads > 1024) {
    return op->emitError("Total number of threads in a thread block ")
           << totalNumThreads
           << " exceeds the limit of 1024 with compilation pipeline "
           << pipelineName;
  }

  // Verify the number of threads in z-dim is 1.
  // 验证z维度的线程数是1。tensorcore的z维度是1，因为专门用来处理matmul操作。
  if (workgroupSize[kDimZ] != 1) {
    return op->emitError("Expected workgroup size in z-dim = 1, but got ")
           << workgroupSize[kDimZ] << " with compilation pipeline "
           << pipelineName;
  }

  // Return success for SIMT/CUDA cores.
  if (pipeline == CodeGenPipeline::LLVMGPUMatmulSimt)
    return success();

  //
  // Additional verification Tensor Core pipelines.
  //

  // Verify that x-dim has multiple of kWarpSize threads or has integer units of
  // warps in x-dim.
  // 验证 x 维度的线程数必须是 warp 大小的倍数
  if (workgroupSize[kDimX] % kWarpSize != 0) {
    return op->emitError("Number of threads in x-dim ")
           << workgroupSize[kDimX] << " is not a multiple of warp size ("
           << kWarpSize
           << ") or integer units of warps in x-dim with compilation pipeline "
           << pipelineName;
  }

  // Number of warps in x, y, and z dim.
  // 计算一个workgroup的warp数目
  // warp一般为32，如果一个workgroup是[64,2,1]，则这里得到[2,2,1]维度的warp
  SmallVector<int64_t> numWarps{workgroupSize[kDimX] / kWarpSize,
                                workgroupSize[kDimY], workgroupSize[kDimZ]};

  // Matrix-multiply problem shape in number of elements in M, N, and K dim.
  // matmulshape为[512, 512, 128]
  // 获取矩阵最本元的MNK参数。
  SmallVector<int64_t> matmulShape{lhsShape[0], rhsShape[1], lhsShape[1]};

  // Warp tile shape in number of elements in M, N, and K dim.
  // Note that num warp in (x, y, z) dim are mapped to problem (M, N, K) dim as:
  // DimY -> ProblemDimM, DimX -> ProblemDimN, DimZ -> ProblemDimK.
  /*
    将线程块（thread block）形状均匀划分给各个 warp。
    注意：注释中说明了 warp 在 (x, y, z) 维度的分布映射到矩阵问题的 (M, N, K) 上，其中：
      y 维度 warp 对应问题的 M；
      x 维度 warp 对应问题的 N；
      z 维度 warp 对应问题的 K；
  */
  // [32/2, 32/2, 16/1] = [16, 16, 16]表征每个warp所要干的工作
  SmallVector<int64_t> warpShape{threadBlockShape[kM] / numWarps[kDimY],
                                 threadBlockShape[kN] / numWarps[kDimX],
                                 threadBlockShape[kK] / numWarps[kDimZ]};

  // Instruction shape in number of elements in M, N, and K dim.
  // 获取tensor core指令形状
  SmallVector<int64_t> instructionShape;
  // f16 和 bf16 类型对应的指令形状为 {16, 16, 16}，而 f32 类型对应 {16, 16, 8} 
  if (failed(getInstructionShape(
          op, pipeline, llvm::cast<ShapedType>(lhsType).getElementType(),
          instructionShape))) {
    return failure();
  }

  // Verify that matmul problem shape can be tiled with the thread block shape.
  // TODO: This check should be relaxed as we allow unaligned matmul shapes.
  // 要求矩阵问题的每个维度（M、N、K）能够被tile设定的维度整除，来决策是否可以做tiling运算
  // 检测[512%32, 512%32, 128%16]
  if (matmulShape[kM] % threadBlockShape[kM] != 0 ||
      matmulShape[kN] % threadBlockShape[kN] != 0 ||
      matmulShape[kK] % threadBlockShape[kK] != 0) {
    return op->emitError("Thread block shape ")
           << threadBlockShape << " cannot be tiled on matmul shape "
           << matmulShape << " with compilation pipeline " << pipelineName;
  }

  // Verify that if warp shape can be tiled using warp-level Tensor core
  // instruction shape.
  // 确保每个 warp tile（即 warpShape）在 M、N、K 维度上均可以被对应的 Tensor Core 指令形状整除。
  // 若不满足，则说明硬件的计算单元（Tensor Core）无法完美地覆盖 warp tile
  // 以f16为例，为[16,16,16]，即每个warp的计算，是否可以生成Tensor Core指令
  if (warpShape[kM] % instructionShape[kM] != 0 ||
      warpShape[kN] % instructionShape[kN] != 0 ||
      warpShape[kK] % instructionShape[kK] != 0) {
    return op->emitError("Tensor Core instruction shape ")
           << instructionShape << " cannot be tiled on warp shape " << warpShape
           << " with compilation pipeline " << pipelineName;
  }

  return success();
}

} // namespace mlir::iree_compiler
