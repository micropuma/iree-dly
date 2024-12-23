// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to emulate 64-bit integer operations with 32-bit
// ones.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/WideIntEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-emulate-i64"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVEMULATEI64PASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {0}", op.getType()));

    auto newOp =
        rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
            op, newResultTy, adaptor.getLayout(), adaptor.getBinding(),
            adaptor.getByteOffset(), adaptor.getDynamicDims(),
            adaptor.getAlignmentAttr(), adaptor.getDescriptorFlagsAttr());
    LLVM_DEBUG(llvm::dbgs()
               << "WideIntegerEmulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

// Tries to flatten `type` to a 1-D vector type. Returns `nullptr` on failure.
static VectorType flattenVectorType(Type type) {
  auto vecTy = llvm::dyn_cast<VectorType>(type);
  if (!vecTy)
    return nullptr;

  if (vecTy.isScalable() || vecTy.getRank() <= 1)
    return nullptr;

  int64_t totalElements = vecTy.getNumElements();
  return VectorType::get(llvm::ArrayRef(totalElements), vecTy.getElementType());
}

// Flattens 2+-D elementwise vector ops into ops over 1-D vectors by inserting
// shape casts, e.g.:
// ```mlir
// %x = arith.muli %a, %b : vector<4x1xi32>
//  ==>
// %flat_a = vector.shape_cast %a : vector<4x1xi32> to vector<4xi32>
// %flat_b = vector.shape_cast %b : vector<4x1xi32> to vector<4xi32>
// %flat_x = arith.muli %flat_a, %flat_b : vector<4xi32>
// %x = vector.shape_cast %flat_x : vector<4xi32> to vector<4x1xi32>
// ```
//
// The shape casts cancel out and can be removed by subsequent canonicalization
// patterns.
struct FlattenElementwisePattern final : RewritePattern {
  FlattenElementwisePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag{}, benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return failure();

    auto newResultTypes = llvm::to_vector_of<Type, 2>(
        llvm::map_range(op->getResultTypes(), flattenVectorType));
    if (llvm::any_of(newResultTypes, [](Type type) { return !type; }))
      return failure();

    Location loc = op->getLoc();

    // Shape cast operands.
    auto operands = llvm::to_vector_of<Value, 2>(op->getOperands());
    for (Value &operand : operands) {
      VectorType newOperandTy = flattenVectorType(operand.getType());
      if (!newOperandTy)
        return failure();

      operand = rewriter.createOrFold<vector::ShapeCastOp>(loc, newOperandTy,
                                                           operand);
    }

    Operation *newOp =
        rewriter.create(loc, op->getName().getIdentifier(), operands,
                        newResultTypes, op->getAttrs());

    // Shape cast results.
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      Value cast = rewriter.create<vector::ShapeCastOp>(
          loc, oldResult.getType(), newResult);
      rewriter.replaceAllUsesWith(oldResult, cast);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static void
populateIreeI64EmulationPatterns(arith::WideIntEmulationConverter &converter,
                                 RewritePatternSet &patterns) {
  patterns.add<ConvertHalInterfaceBindingSubspan>(converter,
                                                  patterns.getContext());
}

static bool supportsI64(FunctionOpInterface op) {
  IREE::GPU::TargetAttr attr = getGPUTargetAttr(op);
  assert(attr && "Missing GPU target");
  return IREE::GPU::bitEnumContainsAll(attr.getWgp().getCompute().getValue(),
                                       IREE::GPU::ComputeBitwidths::Int64);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct SPIRVEmulateI64Pass final
    : impl::SPIRVEmulateI64PassBase<SPIRVEmulateI64Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    if (supportsI64(op))
      return;

    arith::WideIntEmulationConverter typeConverter(32);
    memref::populateMemRefWideIntEmulationConversions(typeConverter);

    MLIRContext *ctx = &getContext();

    // Run the main emulation pass.
    {
      ConversionTarget target(*ctx);
      target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](
                                                     Operation *op) {
        return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
      });
      target.addDynamicallyLegalDialect<
          arith::ArithDialect, func::FuncDialect, IREE::HAL::HALDialect,
          memref::MemRefDialect, vector::VectorDialect>(
          [&typeConverter](Operation *op) {
            bool legal = typeConverter.isLegal(op);
            LLVM_DEBUG(if (!legal) llvm::dbgs()
                       << "WideIntegerEmulation: illegal op: " << *op << "\n");
            return legal;
          });

      RewritePatternSet patterns(ctx);
      arith::populateArithWideIntEmulationPatterns(typeConverter, patterns);
      memref::populateMemRefWideIntEmulationPatterns(typeConverter, patterns);
      populateIreeI64EmulationPatterns(typeConverter, patterns);

      if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
    }

    // Clean up any new 2-D vectors. We need to do it here because later passed
    // may expect any n-D vectors to have been already broken down into 1-D
    // ones.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<FlattenElementwisePattern>(ctx, 100);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);

      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
