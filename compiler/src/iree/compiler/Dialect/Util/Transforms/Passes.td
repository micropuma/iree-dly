// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_PASSES
#define IREE_DIALECT_UTIL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Optimization and cleanup
//===----------------------------------------------------------------------===//

def ApplyPatterns : Pass<"iree-util-apply-patterns", ""> {
  let summary = "Applies some risky/IREE-specific canonicalization patterns.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createApplyPatternsPass()
  }];
}

def CombineInitializers : Pass<"iree-util-combine-initializers", "mlir::ModuleOp"> {
  let summary = "Combines global initializers into one.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createCombineInitializersPass()
  }];
}

def DropCompilerHints : Pass<"iree-util-drop-compiler-hints", ""> {
  let summary = "Deletes operations that have no runtime equivalent.";
  let description = [{
    Deletes operations that have no runtime equivalent and are only
    used in the compiler. This should be performed after all other
    compiler passes.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createDropCompilerHintsPass()
  }];
}

def DumpModule : Pass<"iree-util-dump-module", "mlir::ModuleOp"> {
  let summary = "Dumps the module IR to the given file path.";
  let description = [{
    Dumps the module IR to the given file path in either textual (.mlir) or
    binary (.mlirbc) format. Source locations remain unchanged.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createDumpModulePass()
  }];
}

def FixedPointIterator : Pass<"iree-util-fixed-point-iterator", ""> {
  let summary = "Iterates a sub-pipeline to a fixed point.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFixedPointIteratorPass(
        mlir::OpPassManager("dummy_op"))
  }];
}

def IPO : Pass<"iree-util-ipo", "mlir::ModuleOp"> {
  let summary = "Performs basic inter-procedural optimization.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createIPOPass()
  }];
}

def OptimizeIntArithmetic : Pass<"iree-util-optimize-int-arithmetic", ""> {
  let summary = "Optimizes integer arithmetic using a variety of dataflow analysis and patterns.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createOptimizeIntArithmeticPass()
  }];
}

def PropagateSubranges : Pass<"iree-util-propagate-subranges", "mlir::ModuleOp"> {
  let summary = "Propagates resource subranges across the program.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createPropagateSubrangesPass()
  }];
}

def StripAndSplatConstants :
    Pass<"iree-util-strip-and-splat-constants", "mlir::ModuleOp"> {
  let summary = "Strips constant util.global ops and replaces them with splats.";
  let constructor = "mlir::iree_compiler::IREE::Util::createStripAndSplatConstantsPass()";
}

def StripDebugOps : Pass<"iree-util-strip-debug-ops", ""> {
  let summary = "Strips debug ops, like assertions.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createStripDebugOpsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

def FoldGlobals : Pass<"iree-util-fold-globals", "mlir::ModuleOp"> {
  let summary = "Folds duplicate globals and propagates constants.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFoldGlobalsPass()
  }];
}

def FuseGlobals : Pass<"iree-util-fuse-globals", "mlir::ModuleOp"> {
  let summary = "Fuses correlated globals together.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createFuseGlobalsPass()
  }];
}

def HoistIntoGlobals : Pass<"iree-util-hoist-into-globals", "mlir::ModuleOp"> {
  let summary = "Greedily hoists eligible constant expressions into globals.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createHoistIntoGlobalsPass()
  }];
  let options = [
    Option<"maxSizeIncreaseThreshold", "max-size-increase-threshold", "int64_t",
      /*default=*/"1048576",
      "Maximum byte size increase allowed for constant expr hoisting policy to"
      "allow hoisting. The threshold is 1MB by default.">
  ];
}

def SimplifyGlobalAccesses :
    InterfacePass<"iree-util-simplify-global-accesses", "mlir::CallableOpInterface"> {
  let summary = "Hoists loads and sinks stores to variables to decrease data dependency regions.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createSimplifyGlobalAccessesPass()
  }];
}

//===----------------------------------------------------------------------===//
// Resource Management
//===----------------------------------------------------------------------===//

def ImportResources : Pass<"iree-util-import-resources", ""> {
  let summary = "Imports IR with arbitrary large-data into resources that IREE can manage efficiently";
  let description = [{
    MLIR has many interesting ways to store large constants, most of which
    derive from *ElementsAttr. Given the uniquing/inline behavior, this exacts
    very large runtime and memory overhead costs.

    This is a temporary pass to convert a majority of the legacy
    DenseElementsAttr attributes to DenseResourceElementsAttr. Ideally this
    is done at the source (frontend), but this pass is provided to aid
    transition and testing by doing a manual conversion with iree-opt.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createImportResourcesPass()
  }];
}

//===----------------------------------------------------------------------===//
// Debug/test passes
//===----------------------------------------------------------------------===//

def AnnotateOpOrdinals : Pass<"iree-util-annotate-op-ordinals", "mlir::ModuleOp"> {
  let summary = "Annotates ops with globally unique IDs for debugging.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createAnnotateOpOrdinalsPass()
  }];
}

def TestConversion : Pass<"iree-util-test-conversion", "mlir::ModuleOp"> {
  let summary = "Tests util dialect conversion patterns.";
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createTestConversionPass()
  }];
}

def TestFloatRangeAnalysis : Pass<"iree-util-test-float-range-analysis", ""> {
  let summary = "Tests floating point range analysis.";
  let description = [{
    Tests floating point range analysis by evaluating any
    'iree_unregistered.test_fprange' op and setting the results on an attribute.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Util::createTestFloatRangeAnalysisPass()
  }];
}

#endif  // IREE_DIALECT_UTIL_PASSES
