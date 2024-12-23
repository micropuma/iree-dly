// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_UTILTOVM_PATTERNS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_UTILTOVM_PATTERNS_H_

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends IREE special hint ops to VM dialect patterns.
void populateUtilToVMPatterns(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter,
                              ImportTable &importTable,
                              RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_UTILTOVM_PATTERNS_H_
