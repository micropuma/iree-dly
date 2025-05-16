// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-tile-and-distribute"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUTILEANDDISTRIBUTEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

/// Tiles to workgroup level. Workgroup tiling is done at the flow level but we
/// may have extra tiling for the reduction dimension. Therefore we tile again
/// without distributing.
static LogicalResult tileReductionLoops(mlir::FunctionOpInterface funcOp) {
  auto tileSizesFn = [](OpBuilder &builder,
                        Operation *op) -> SmallVector<OpFoldResult> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);      // 获取可分块的循环组

    // Returns the loop IDs (0 being outermost) that are partitionable.
    // 
    // If `maxNumPartitionedLoops` is passed the size of the vector returned
    // is always lesser than the value.
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);                // 获取已在workgroup中分块的循环         
    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(op->getContext(), getTileSizes(op, 0));
    auto zeroAttr = builder.getIndexAttr(0);
    for (unsigned depth : partitionedLoops) {             
      if (depth < tileSizes.size()) {                // 将已分配到工作组的并行循环平铺尺寸设为0（跳过）
        tileSizes[depth] = zeroAttr;
      }
    }

    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();                       //  ​​确保分块尺寸向量（tileSizes）的长度与操作的循环维度数量一致​​，并为未明确指定分块大小的维度设置默认不分块（即分块大小为0）       
    tileSizes.resize(numLoops, zeroAttr);
    return tileSizes;
  };

  auto tilingOptions =
      scf::SCFTilingOptions().setTileSizeComputationFunction(tileSizesFn);

  MLIRContext *context = funcOp.getContext();
  LinalgTransformationFilter filter(                                                                   //  处理源操作中带有workgroup memory标记的操作
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));                        // 处理好后的打上tilemarker
  filter.setMatchByDefault();

  return tileLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

/// 这个pass的作用是按照workgroup size来做tiling。
static LogicalResult tileToSerialLoops(mlir::FunctionOpInterface funcOp) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    if (failed(tileReductionLoops(funcOp))) {
      return failure();
    }
  }

  {       // 一些tiling的善后工作
    RewritePatternSet wgTilingCanonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
    populateAffineMinSCFCanonicalizationPattern(
        wgTilingCanonicalizationPatterns);
    scf::populateSCFForLoopCanonicalizationPatterns(
        wgTilingCanonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
      return failure();
    }
    return success();
  }
}

/// Return the tile size associated to one thread or warp based on the number of
/// element in the group.
static SmallVector<Value>
calculateDistributedTileSize(ArrayRef<int64_t> numElements, OpBuilder &builder,
                             Operation *operation) {
  SmallVector<int64_t> blockTileSize = getTileSizes(operation, 0);
  SmallVector<Value> tileSizesVal;
  // Use partitionedLoop to know what loop needs to be distributed.
  auto interfaceOp = cast<PartitionableLoopsInterface>(operation);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    return tileSizesVal;
  }
  auto zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
  tileSizesVal.resize(
      cast<TilingInterface>(operation).getLoopIteratorTypes().size(), zero);

  // partitionedLoops contains the dimensions we want to distribute.
  // We are distributing them in order onto the different workgroup
  // dimensions.
  SmallVector<int64_t> distributedDim(numElements.begin(), numElements.end());
  distributedDim.resize(partitionedLoops.size());
  unsigned idIdx = 0;
  std::reverse(distributedDim.begin(), distributedDim.end());
  for (unsigned depth : partitionedLoops) {
    if (depth >= blockTileSize.size())
      continue;
    tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
        operation->getLoc(),
        llvm::divideCeil(blockTileSize[depth], distributedDim[idIdx++]));
    if (idIdx == kNumMaxParallelDims)
      break;
  }
  return tileSizesVal;
}

/// Tiles to warp.
static LogicalResult tileToWarp(mlir::FunctionOpInterface funcOp,
                                SmallVectorImpl<int64_t> &workgroupSize) {
  std::array<int64_t, 3> warpPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [warpPerWorkgroup](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(warpPerWorkgroup, builder,
                                            operation);
      };
  auto getWarpProcInfoFn = [warpPerWorkgroup](
                               OpBuilder &builder, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getSubgroupIdsAndCounts(builder, loc, /*warpSize=*/32u,
                                   parallelLoopRanges.size(), warpPerWorkgroup);
  };
  linalg::LinalgLoopDistributionOptions warpDistributionOptions;
  warpDistributionOptions.procInfo = getWarpProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(warpDistributionOptions);
  MLIRContext *context = funcOp.getContext();
  LinalgTransformationFilter filter(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  filter.setMatchByDefault();
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

/// Patterns for thread level tiling.
static LogicalResult tileToInvocation(mlir::FunctionOpInterface funcOp,
                                      SmallVectorImpl<int64_t> &workgroupSize) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(workgroupSize, builder, operation);    // 计算tile size: [64/32, 2/1, 1/1] = [2, 2, 1]
      };
  auto getThreadProcInfoFn =                                                            // 显示计算生成thread id，不断向GPU的kernel表达靠近
      [&workgroupSize](OpBuilder &builder, Location loc,
                       ArrayRef<Range> parallelLoopRanges) {
        return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                        workgroupSize);
      };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;                             // 设定thread的分布选项，设定procInfo
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  MLIRContext *context = funcOp.getContext();
  LinalgTransformationFilter f(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  f.addFilter([](Operation *op) {
     // FFT doesn't support second level of tiling yet.
     return success(!isa<IREE::LinalgExt::FftOp>(op));
   }).setMatchByDefault();

  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, f);
}

namespace {
class LLVMGPUTileAndDistributePass final
    : public impl::LLVMGPUTileAndDistributePassBase<
          LLVMGPUTileAndDistributePass> {
private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

public:
  using impl::LLVMGPUTileAndDistributePassBase<
      LLVMGPUTileAndDistributePass>::LLVMGPUTileAndDistributePassBase;
  LLVMGPUTileAndDistributePass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }

  // debug1：tensor core
  void runOnOperation() override {   
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // Promote C matrix and propagate the potential  fill producer into the temp
    // allocation. This needs to be done before reduction tiling.
    {
      RewritePatternSet promotionPatterns(&getContext());

      // Adds patterns for promoting Linalg contract op's operands to use GPU shared
      // memory.
      populateContractPromotionPatterns(promotionPatterns, {2});
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      propagateSharedMemoryCopy(funcOp);    // 显示取消拷贝入共享内存操作，改为直接写入共享内存/或是内联入linalg.genericOp的方式
    }

    // Tile again at the workgroup level since reduction dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    if (failed(tileToSerialLoops(funcOp))) {         // 将k维度（规约维度）按照tilesize进行分块
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile reductions:";
      funcOp.dump();
    });

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      funcOp.emitOpError("expected workgroup size to be set on the lowering "
                         "config for the function");
      return signalPassFailure();
    }

    SmallVector<int64_t> workgroupSize = maybeWorkgroupSize.value();
    int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    // Only promote to workgroup size if there are multiple warps.
    if (flatWorkgroupSize > kWarpSize) {                             // 以[64,2,1]为例，flatWorkgroupSize=128时，表示有2x2个warp，所以切分成warp是有必要的
      RewritePatternSet promotionPatterns(&getContext());

      populateContractPromotionPatterns(promotionPatterns, {0, 1});     // 0和1表示对A/B矩阵进行提升

      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      // Insert barriers before and after copies to workgroup memory.
      insertBarriersAroundSharedMemoryCopy(funcOp);
    }

    {
      RewritePatternSet promotionCanonicalization =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(promotionCanonicalization)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After promotion:";
      funcOp.dump();
    });

    if (distributeToWarp) {
      // Apply last level of tiling and distribute to warps.
      if (failed(tileToWarp(funcOp, workgroupSize))) {
        return signalPassFailure();
      }

    } else {
      // Apply last level of tiling and distribute to threads.
      if (failed(tileToInvocation(funcOp, workgroupSize))) {
        return signalPassFailure();
      }
    }
    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile and distribute to threads:";
      funcOp.dump();
    });
  }
};
} // namespace

/*
LLVMGPUTileAndDistribute Pass的工作流程：
  * 提升C到共享内存：减少全局内存访问。
  * K维度分块：分解计算为小块，适配寄存器和共享内存。
  * 提升A/B到共享内存（可选）：进一步减少全局内存依赖。
  * Warp级别分块：映射计算到GPU线程层次，最大化并行度。
性能优化核心：
  * 数据局部性：通过共享内存缓存减少全局内存访问。
  * 并行度：Workgroup和Warp两级并行，充分利用GPU线程资源。
  * 硬件适配：分块尺寸匹配Tensor Core指令要求（如16x16x16）。
*/
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUTileAndDistributePass(bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileAndDistributePass>(distributeToWarp);

} // namespace mlir::iree_compiler
