// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::InputConversion {

#define GEN_PASS_DEF_SANITIZEMODULENAMESPASS
#include "iree/compiler/InputConversion/Common/Passes.h.inc"

namespace {

// 这个pass做的工作相当简单：
// 将变量名字对于IREE的规则做适配。
// std::string sanitizeSymbolName(StringRef name) {
//   std::string result;
//   result.reserve(name.size());
//   for (size_t i = 0; i < name.size(); ++i) {
//     char c = name[i];
//     if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
//           (c >= '0' && c <= '9') || c == '_')) {
//       c = '_';
//     }
//     result.push_back(c);
//   }
//   return result;
// }
class SanitizeModuleNamesPass final
    : public impl::SanitizeModuleNamesPassBase<SanitizeModuleNamesPass> {
public:
  void runOnOperation() override {
    // MLIR identifiers must match this regex:
    //   (letter|[_]) (letter|digit|[_$.])*
    // https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords
    //
    // IREE VM modules use the `.` (period) character for namespacing, so
    // replace any occurrences of `.` with `_`.

    auto moduleOp = getOperation();
    auto optionalName = moduleOp.getName();
    if (!optionalName.has_value())
      return;
    auto name = optionalName.value();

    moduleOp.setName(sanitizeSymbolName(name));
  }
};

} // namespace
} // namespace mlir::iree_compiler::InputConversion
