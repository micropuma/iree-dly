// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> clDispatchTransformFileName(
    "iree-dispatch-creation-dispatch-use-transform-dialect",
    llvm::cl::desc("MLIR file containing a top-level module that specifies "
                   "the transformations to apply to form dispatch regions."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clDetensoring(
    "iree-dispatch-creation-enable-detensoring",
    llvm::cl::desc(
        "Enable changing of tensor operations into scalar operations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableElementWiseFuseMultiReduction(
    "iree-dispatch-creation-element-wise-fuse-multi-reduction",
    llvm::cl::desc("Enable element-wise fusion of multi-reduction loop ops."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgConsumerOps(
    "iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgProducerOps(
    "iree-dispatch-creation-enable-fuse-padding-into-linalg-producer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops."),
    llvm::cl::init(false));

static llvm::cl::opt<int> clPadFactor(
    "iree-dispatch-creation-pad-factor",
    llvm::cl::desc("Provides padding size hints that will be attached to "
                   "encodings. This only affects the experimental data tiling "
                   "path in DispatchCreation with "
                   "iree-dispatch-creation-experimental-data-tiling."),
    llvm::cl::init(32));

static llvm::cl::opt<bool> clEnablePadHandling(
    "iree-flow-enable-pad-handling",
    llvm::cl::desc("Enable native handling of tensor.pad operations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFuseHorizontalContractions(
    "iree-dispatch-creation-enable-fuse-horizontal-contractions",
    llvm::cl::desc(
        "Enables horizontal fusion of contractions with one common operand"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clCollapseReductionDims(
    "iree-dispatch-creation-collapse-reduction-dims",
    llvm::cl::desc("Enable collapsing of reduction dims"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clEnableFuseMultiUse("iree-dispatch-creation-fuse-multi-use",
                         llvm::cl::desc("Fuse multi-use ops."),
                         llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableAggressiveFusion(
    "iree-dispatch-creation-enable-aggressive-fusion",
    llvm::cl::desc("Aggressive fusion opportunities that are behind a flag "
                   "since all backends dont support it yet"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableDataTiling(
    "iree-dispatch-creation-experimental-data-tiling",
    llvm::cl::desc("Enable data-tiling at flow level, i.e., it sets encodings "
                   "in dispatch regions, hoist them out of region, and enables "
                   "fusion for the set_encodings. This is still an "
                   "experimental path. The current main data tiling path is "
                   "iree-opt-data-tiling, which is on by default. To use this "
                   "path, --iree-opt-data-tiling=false must be set as wells"),
    llvm::cl::init(false));

namespace mlir::iree_compiler::DispatchCreation {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//
using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

// CleanupPatterns是在运行每一组pass之后用于清理无用代码的passs。
static void addCleanupPatterns(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      // Standard MLIR cleanup.
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // Simplify util.global accesses; this can help with data flow tracking as
      // redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass)

      // Aggressive cleanup.
      .addPass(IREE::Util::createApplyPatternsPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());

  // Large IPO pass. Note that this can introduce a significant amount of
  // duplication/inlined constants and we'll want to ensure we're running
  // cleanup again after (this entire set of patterns is run in a fixed-point
  // iteration to do that).
  passManager.addPass(IREE::Util::createIPOPass());
}

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// 做dispatch creation步骤的前期准备
void addDispatchRegionCreationPreprocessingPasses(OpPassManager &passManager) {
  // 从pass的用法上，一般有如下结构：
  // 首先是我们想要做的pass优化
  // 然后是Canonicalize以及CSE等通用优化pass
  // 1. Do some simple elementwise op fusion. This could be skipped,
  //    but could reduce the surface area of ops to handle later.
  FunctionLikeNest(passManager)
      .addPass([]() {
        return DispatchCreation::createElementwiseOpFusionPass(
            ElementwiseOpFusionPassOptions{
                clEnableElementWiseFuseMultiReduction});
      })
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 2. Bubble up expand_shape ops (or sink collapse_shape ops) to get
      //    elementwise operation into higher dimensions for more fusion
      //    opportunities.
      // （1）如果某个expand_shape在elementwise操作之后，则提升这个expand_shape操作。
      // （2）如果某个collapse操作在elementwise操作之前，则下降这个collapse操作。
      .addPass(DispatchCreation::createBubbleUpExpandShapesPass)
      .addPass(DispatchCreation::createBubbleUpExtractSlicesPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 3. Perform elementwise operation fusion again (now with higher
      //    dimensionality).
      .addPass([]() {
        return DispatchCreation::createElementwiseOpFusionPass(
            ElementwiseOpFusionPassOptions{
                clEnableElementWiseFuseMultiReduction});
      })
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 4. After elementwise operation fusion sink reshapes that block
      //    producer-consumer fusion.
      // 将reshapes op下降，以免切断了producer-consumer模式
      .addPass(DispatchCreation::createSinkReshapesPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // 做水平方向上的收缩操作，目前理解是：
  // A = BxC，D = ExF，那么可以把两者融合为一个矩阵计算？
  if (clEnableFuseHorizontalContractions) {
    FunctionLikeNest(passManager)
        .addPass(createFuseHorizontalContractionsPass)
        .addPass(mlir::createCanonicalizerPass)
        .addPass(mlir::createCSEPass);
  }

  FunctionLikeNest(passManager)
      // 5. After all the reshape propagations, fuse elementwise operations
      //    even if the producer has multiple uses.
      .addPass(DispatchCreation::createFuseMultiUseElementwiseProducerPass)

      // 6. Some more "post elementwise fusion passes".
      //    a. Detensorize.
      //       TODO: This is probably not in the right place.
      // 这几个pass的写法，都是只有clDetensoring为true，才执行pass
      .addPredicatedPass(clDetensoring,
                         [&]() { return mlir::createLinalgDetensorizePass(); })
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      //    b. For ops with multiple reduction dimensions, collapse the
      //       reduction dimension.
      //       TODO: This pass is only needed till all backends can handle
      //       multiple reduction dimensions.
      // 这个pass的作用是针对hardware不支持多维度归约情况（reduce-max），
      // 通过collapse操作，将维度下降到hardware支持归约的维度情况。
      .addPredicatedPass(
          clCollapseReductionDims,
          DispatchCreation::createCollapseReductionDimensionsPass)

      //     c. Split reduction operations into parallel and reduction, i.e
      //        .
      .addPass(DispatchCreation::createSplitReductionPass)

      //     d. Transpose generic ops to
      //        - help with dispatch region formation.
      //        - move reduction iterators to be innermost.
      // 对于Linalg.generic op做进一步的处理，方便后续的dispatch生成。
      // 该pass重点是将reduction iteration type放在循环的最内层。
      .addPass(DispatchCreation::createTransposeGenericOpsPass);
}

// Pipeline to first create `flow.dispatch.region` ops and then lower to
// `flow.dispatch.workgroup` ops.
// Dispatch的核心代码段
static void addDispatchRegionCreationPasses(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      // Only want use the transform dialect for some dispatch regions and let
      // the FormDispatchRegions handle the rest. This only moves the root
      // compute op into the dispatch region, so that we can run additional
      // transformations afterwards with a simple region and without bothering
      // producers.
      // 理解dispatch机制，一定理解这个pass。
      // 逻辑是先找到合适的dispatch的root，然后提到dispatch中，经过后续的pass逐渐形成dispatch。
      // 下面是.td中的描述：
      // Pass to perform dispatch of Linalg on tensor ops by using the transform
      // dialect. Dispatch regions are created as specified by the transform module
      // that is parsed from `transformSpecPath`.
      // 这个pass一般不操作
      .addPredicatedPass(
          !clDispatchTransformFileName.empty(),
          [&]() {
            DispatchWithTransformDialectPassOptions options;
            options.transformSpecPath = clDispatchTransformFileName;

            // todo: 学习transform dialect，来从tensor dialect中
            // 找到dispatch的anchor point。
            // 这个pass默认是不开启的，这个pass可以方便用户自定义transform dialect，
            // 来自定义dispatch生成模型。
            return createDispatchWithTransformDialectPass(options);
          })
      // Create dispatches for scalar operations as roots
      .addPass(DispatchCreation::createFormScalarDispatchesPass)
      // Create `flow.dispatch.region` centered around a root and fuse with
      // producers and consumers.
      // 比较有意思的pass，融合生产者和消费者
      // 先前的FormScalar没有处理<tensor>，只处理<tensor1xf>等scalar tensor
      // 这个重点关注Linalg op
      .addPass([&]() {
        return DispatchCreation::createFormDispatchRegionsPass(
            FormDispatchRegionsPassOptions{
                clEnableAggressiveFusion,
                clEnableFusePaddingIntoLinalgConsumerOps,
                clEnableFusePaddingIntoLinalgProducerOps});
      })
      // Clone all producers into the dispatch region to perpare for being
      // isolated from above. This enables running additional transformations
      // afterwards that would need the full dispatch content but don't want to
      // handle explicit captures as materialized as dispatch workgroup operands
      // and block arguments.
      // 下面是详细的description：
      // let description = [{
      // Pass to clone into dispatch regions producers of values used in the dispatch
      // regions but defined in the above. This prepares the dispatch regions for
      // converting to dispatch workgroups with explicit captures.
      // }];
      .addPass(DispatchCreation::createCloneProducersIntoDispatchRegionsPass);

  // 这是实验性的data tiling pass
  // Experimental data tiling path. The intent of this path is to set encodings
  // after fusion decisions have already been made, so encodings can be
  // separated from compiler fusion decisions.
  if (clEnableDataTiling) {
    SetEncodingPassOptions options{clPadFactor};
    FunctionLikeNest(passManager)
        // Set encodings on all eligible ops. All ops should be in compiler
        // formed dispatch regions, so encodings will be placed inside of the
        // dispatch regions with the data-tiled op.
        .addPass([&]() { return createSetEncodingPass(options); })
        // SetEncodingOps should not be in the same dispatch as the data-tiled
        // op, so hoist them out of their current dispatch regions. Also, bubble
        // SetEncodingOps through special operations like bit-extending ops and
        // broadcasting ops.
        .addPass(DispatchCreation::createHoistEncodingOpsPass)
        // After SetEncodingOps are hoisted, try to fuse them with their
        // producer dispatches to try to hide packing costs.
        .addPass(
            DispatchCreation::createFuseEncodingOpsIntoDispatchRegionsPass);
  }
  FunctionLikeNest(passManager)
      // Collapse dimensions of linalg Ops.
      .addPass(DispatchCreation::createCollapseDimensionsPass);
}

// todo：需要继续分析
// Apply preprocessing and form dispatch regions
// 生成dispatch regions，方便后续的Flow（workload和dispatch分离） transformation。
void buildDispatchCreationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {

  // Inject tensor tracing early as we need to have the tracers in the IR
  // prior to dispatch region formation where we may lose access to them.
  FunctionLikeNest(passManager)
      .addPass(IREE::Flow::createInjectTensorTracingPass);

  // Transform pad operations into linalg.fill + tensor.insert_slice.
  // This is a WAR for not having native pad handling.
  // 将pad操作变成熟悉的linalg.fill和tensor.insert_slice操作。
  if (!clEnablePadHandling && !clEnableFusePaddingIntoLinalgProducerOps) {
    passManager.addPass(
        DispatchCreation::createTensorPadToTensorInsertSlicePass(
            TensorPadToTensorInsertSlicePassOptions{
                /*skipSingleLinalgOpUses=*/
                clEnableFusePaddingIntoLinalgConsumerOps}));
  }

  {
    // We run these under a fixed-point iteration such that we can perform
    // inter-procedural, intra-procedural, and canonicalization as separably
    // verifiable/reusable passes. IPO will fold duplicate arguments/results
    // and inline constants to allow the local optimizations to work more
    // effectively.
    // 这是过程间的pass调用。
    OpPassManager ipoPipeline(mlir::ModuleOp::getOperationName());

    // IPO and other cleanups.
    addCleanupPatterns(ipoPipeline);

    // Run fixed-point iteration on the IPO pipeline.
    passManager.addPass(
        IREE::Util::createFixedPointIteratorPass(std::move(ipoPipeline)));
  }

  FunctionLikeNest(passManager)
      // Preprocess the input to a form more amenable for fusion.
      // 理解一下这个FusionPreprocessingPass的作用。
      .addPass(DispatchCreation::createFusionPreprocessingPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // todo: 后续可以重点分析
  addDispatchRegionCreationPreprocessingPasses(passManager);
  addDispatchRegionCreationPasses(passManager);

  // 划分好dispatch后，将dispatch变成workgroups
  // 与AIE的tile划分十分类似
  FunctionLikeNest(passManager)
      .addPass(DispatchCreation::createConvertDispatchRegionsToWorkgroupsPass)

      // Convert tensor operations to flow.tensor ops.
      // - Convert extract/insert slice to flow update ops when the tensor op
      // acts as a contiguous view of the tensor
      // - Apply tensor -> flow patterns
      // 生成flow的pass，也是重点。这个pass值得学习，学习如何提出
      // 从tensor pattern中得到flow patterns
      .addPass(DispatchCreation::createConvertTensorToFlowPass)
      .addPass(createCSEPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      /// Creates the workgroup count region where the materialized computation
      /// is derived as a program slice of the body of the dispatch. This method
      /// - Computes the `workload` to use for the `workgroupsOp`, which are
      ///   derived from the values captured by the `workgroupsOp`.
      /// - Populates the workgroup count region for this with the placeholder
      ///   op `flow.dispatch.workgroups_count_from_body_slice`. This op is
      ///   resolved in the backends into the actual workgroup count
      ///   computation.
      /// - To correlate back to the captured workload,
      /// `flow.dispatch.workload.ordinal`
      ///   to map the captured operand to the position in the workload list.
      .addPass(
          DispatchCreation::createMaterializeDefaultWorkgroupCountRegionPass);
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/DispatchCreation/Passes.h.inc" // IWYU pragma: keep
} // namespace

void registerDispatchCreationPasses() {
  // Generated from Passes.td
  registerPasses();
}

void registerDispatchCreationPipelines() {
  PassPipelineRegistration<TransformOptions> dispatchCreationPipeline(
      "iree-dispatch-creation-pipeline",
      "Flag used to run passes that form dispatch regions",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildDispatchCreationPassPipeline(passManager, transformOptions);
      });

  PassPipelineRegistration<> dispatchCreationPreprocessingPipeline(
      "iree-dispatch-creation-preprocessing-pipeline",
      "Flag used to run preprocessing passes that run passes before dispatch "
      "region formation. Used only for testing",
      [](OpPassManager &passManager) {
        addDispatchRegionCreationPreprocessingPasses(passManager);
      });
}

} // namespace mlir::iree_compiler::DispatchCreation
