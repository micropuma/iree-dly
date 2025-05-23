// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MEMREFCOPYTOLINALGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct MemrefCopyOpToLinalg : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // 核心逻辑特别简单：用linalg.copy替换memref.copy
    Operation *linalgCopy =
        createLinalgCopyOp(rewriter, copyOp.getLoc(), copyOp.getSource(),
                           copyOp.getTarget(), copyOp->getAttrs());
    if (!linalgCopy)
      return failure();
    rewriter.replaceOp(copyOp, linalgCopy->getResults());
    return success();
  }
};

struct MemrefCopyToLinalgPass final
    : impl::MemrefCopyToLinalgPassBase<MemrefCopyToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    // 注册memref copy到linalg copy的pattern
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<MemrefCopyOpToLinalg>(context);

    // 应用pattern
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
