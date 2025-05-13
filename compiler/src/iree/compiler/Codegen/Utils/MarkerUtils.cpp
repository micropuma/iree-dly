// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
  if (f) {
    filters.push_back(f);
  }
}

LogicalResult LinalgTransformationFilter::checkAndNotify(RewriterBase &rewriter,
                                                         Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); })) {        // 给operation应用每个FilterFunciton
    return failure();
  }

  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty() || matchByDefault) {
      return success();
    }

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction) {
    if (attr.getValue() == filter) {                    // 在tilek这个场景下，filter是workgroup_memory，即这个operation在filter中是需要处理的
      return success();
    }
  }

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void LinalgTransformationFilter::replaceLinalgTransformationFilter(
    RewriterBase &rewriter, Operation *op) const {
  // 如果指定替换，则替换kLinalgTransformMarker的attr为新的replacement，否则单纯删除kLinalgTransformMarker
  if (replacement.has_value()) {
    op->setAttr(LinalgTransforms::kLinalgTransformMarker, replacement.value());
  } else {
    op->removeAttr(
        rewriter.getStringAttr(LinalgTransforms::kLinalgTransformMarker));
  }
}

bool LinalgTransformationFilter::hasReplacementFilter(Operation *op) const {
  if (!replacement) {
    return false;
  }
  auto attr = dyn_cast<StringAttr>(
      op->getAttr(LinalgTransforms::kLinalgTransformMarker));
  return attr && attr == *replacement;
}

struct VectorTransforms {
  static const StringLiteral kVectorTransformMarker;
};
const StringLiteral VectorTransforms::kVectorTransformMarker =
    "__internal_vector_transform__";

StringRef getFusedMarker() { return "fused_numprocs_ge_numiters"; }

StringRef getWorkgroupKTiledMarker() { return "workgroup_k_tiled"; }

StringRef getWorkgroupL1TileMarker() { return "workgroup_l1_tile"; }

StringRef getWorkgroupMemoryMarker() { return "workgroup_memory"; }

StringRef getWorkgroupNumItemsGENumItersMarker() {
  return "workgroup_numprocs_ge_numiters";
}

StringRef getWorkgroupMemoryNumItemsGENumItersMarker() {
  return "workgroup_memory_numprocs_ge_numiters";
}

StringRef getCopyToWorkgroupMemoryMarker() {
  return "copy_to_workgroup_memory";
}

StringRef getTileReductionMarker() { return "tile_reduction"; }

StringRef getVectorizeMarker() { return "vectorize"; }

StringRef getDeleteMarker() { return "delete"; }

StringRef getMarkerOrNull(Operation *op) {
  StringAttr attr =
      op->getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker);
  if (!attr)
    return "";
  return attr.getValue();
}

// memref.copy %subview_2, %subview_1 {__internal_linalg_transform__ = "copy_to_workgroup_memory"}
// 为例，copy_to_workgroup_memory是一个kLinalgTransformMarker的值 
bool hasMarker(Operation *op, ArrayRef<StringRef> marker) {          // 辅助函数，判断一个operation是否有某个marker
  StringAttr attr =
      op->getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker);
  return attr && (marker.empty() ||
                  llvm::any_of(marker, [&attr](StringRef markerValue) {     // llvm::any_of底层是std::any_of(marker.begin(), marker.end(), lambad函数)
                    return attr.getValue() == markerValue;
                  }));
}

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(LinalgTransforms::kLinalgTransformMarker,
              StringAttr::get(op->getContext(), marker));
}

constexpr StringLiteral kUnrollLoopName = "unroll_loop";
void setLoopUnrollMarker(Operation *op) {
  op->setAttr(kUnrollLoopName, UnitAttr::get(op->getContext()));
}

Attribute getLoopUnrollMarker(Operation *op) {
  return op->getAttr(kUnrollLoopName);
}

void removeLoopUnrollMarker(Operation *op) { op->removeAttr(kUnrollLoopName); }

} // namespace mlir::iree_compiler
