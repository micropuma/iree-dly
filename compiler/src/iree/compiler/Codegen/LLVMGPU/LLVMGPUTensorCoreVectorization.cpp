// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-tensorcore-vectorization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUTENSORCOREVECTORIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void vectorizeLinalgOps(mlir::FunctionOpInterface funcOp) {
  MLIRContext *context = funcOp.getContext();
  IRRewriter rewriter(context);

  // 构建linalg 变化的过滤函数
  // 关键过滤依赖getVectorizeMarker()函数
  LinalgTransformationFilter f(StringAttr::get(context, getVectorizeMarker()));

  funcOp.walk([&](Operation *op) {
    // 1.使用过滤器，查看是否匹配
    // 2.linalg::FillOp, linalg::GenericOp, linalg::ContractionOpInterface是必须匹配的项
    if (failed(f.checkAndNotify(rewriter, op)) ||
        !isa<linalg::FillOp, linalg::GenericOp, linalg::ContractionOpInterface>(
            op)) {
      // 如果上述要求都不满足，则跳过
      return WalkResult::advance();
    }
    // 针对该operation进行vectorize操作
    (void)linalg::vectorize(rewriter, op);
    return WalkResult::advance();
  });
}

// 将vector进行展开，这里的展开是指将vector转换为native tensor core operations
static void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                         bool useMmaSyncShape) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  auto getNativeShape = [useMmaSyncShape](Operation *op) {
    if (useMmaSyncShape)
      // 强调warp之间的协同，一般更复杂
      return getMmaNativeVectorSize(op);
    // 性能更优，但是warp之间的协同更简单
    return getWmmaNativeVectorSize(op);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getNativeShape)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

namespace {
class LLVMGPUTensorCoreVectorizationPass final
    : public impl::LLVMGPUTensorCoreVectorizationPassBase<
          LLVMGPUTensorCoreVectorizationPass> {
public:
  using impl::LLVMGPUTensorCoreVectorizationPassBase<
      LLVMGPUTensorCoreVectorizationPass>::
      LLVMGPUTensorCoreVectorizationPassBase;
  explicit LLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType tensorCoreType)
      : tensorCoreType(tensorCoreType) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  // %8 = vector.transfer_read %subview_8[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
  // %9 = vector.transfer_read %subview_9[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
  // %10 = vector.transfer_read %subview_10[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
  // %11 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %10 : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
  // 上述codes是vectorization的核心代码，vector.contract是linalg的操作，%8, %9, %10是vector.transfer_read的结果，%11是vector.contract的结果
  void runOnOperation() override {
    auto funcOp = getOperation();
    LLVM_DEBUG({
      llvm::dbgs() << "LLVMGPUTensorCoreVectorizationPass runOnOperation():\n";
      funcOp->dump();
    });

    MLIRContext *context = &getContext();
    {
      // Step 1(a). Vectorize (linalg to vector).
      vectorizeLinalgOps(funcOp);
      RewritePatternSet contractionPatterns(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractionPatterns);
      /// Collect patterns to convert reduction op to vector.contract and fold
      /// transpose/broadcast ops into the contract.
      vector::populateVectorReductionToContractPatterns(contractionPatterns);
      /// Patterns that remove redundant Vector Ops by re-ordering them with
      /// e.g. elementwise Ops:
      /// ```
      /// %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
      /// %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
      /// %r = arith.addf %at, %bt : vector<2x4xf32>
      /// ```
      /// gets converted to:
      /// ```
      /// %0 = arith.addf %a, %b : vector<4x2xf32>
      /// %r = vector.transpose %0, [1, 0] : vector<2x4xf32>
      /// ```
      vector::populateSinkVectorOpsPatterns(contractionPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(contractionPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter populateVectorizationPatterns:\n";
        funcOp->dump();
      });

      // Step 1(b). Fold arithmetic extensions into vector contraction ops.
      // Linalg to vector conversion introduces arithmetic extensions on the
      // operands of vector contraction ops for mixed precision computation.
      // This pattern folds the arithmetic extensions into the vector.contract.
      // 将算术扩展折叠到vector.contract中，因为tensor core适配混合精度，可能后续并不需要
      // 这些算术扩展
      RewritePatternSet foldArithExtPatterns(context);
      vector::populateFoldArithExtensionPatterns(foldArithExtPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(foldArithExtPatterns)))) {
        return signalPassFailure();
      }

      // Step 2. Fold consumer add ops into the contraction op itself.
      // 很典型的producer consumer模式，将consumer add ops折叠到contraction op中
      // 以减少内存访问
      // TODO-tensor-core:需要继续深入contract op的实现
      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);

      // tensor core的MMA计算不支持transpose op，但是MMA的read算子支持按照transpose的方式transfer tensor
      populateCombineVectorTransferReadBroadcastPatterns(
          canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs()
            << "\nAfter populateCombineVectorTransferReadBroadcastPatterns:\n";
        funcOp->dump();
      });

      // Step 3. Prepare vector operations to be lowered to native tensor core
      // operations (nvgpu.mmasync, nvgpu.ldmatrix). 
      if (tensorCoreType == GPUTensorCoreType::MMA_SYNC) {
        RewritePatternSet vectorContractPatterns(funcOp.getContext());
        mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
            vectorContractPatterns);

        /// Patterns to transform vector ops into a canonical form to convert to MMA
        /// matrix operations. If `useNvGpu` is true, then the patterns will populated
        /// will prepare for conversion to `nvgpu` mma operations rather than the `gpu`
        /// dialect WMMA operations.
        mlir::populatePrepareVectorToMMAPatterns(vectorContractPatterns,
                                                 /*useMMASync=*/true);
        if (failed(applyPatternsAndFoldGreedily(
                getOperation(), std::move(vectorContractPatterns)))) {
          return signalPassFailure();
        }
      }
      LLVM_DEBUG({
        llvm::dbgs()
            << "\nAfter populateCastAwayVectorLeadingOneDimPatterns and "
               "populatePrepareVectorToMMAPatterns:\n";
        funcOp->dump();
      });

      bool useMmaSyncShape = tensorCoreType == GPUTensorCoreType::MMA_SYNC;
      // Step 4. Break and unroll warp tile size to native math and load sizes.
      // 拆分成warp tile size 
      // TODO-tensor-core: 这块详细debug
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns, useMmaSyncShape);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter populateVectorUnrollPattern:\n";
        funcOp->dump();
      });
    }
  }

private:
  GPUTensorCoreType tensorCoreType;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType tensorCoreType) {
  return std::make_unique<LLVMGPUTensorCoreVectorizationPass>(tensorCoreType);
}

} // namespace mlir::iree_compiler
