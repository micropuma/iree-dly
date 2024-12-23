// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPUPREPAREFORCHAINEDMATMULPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

using VectorValue = TypedValue<VectorType>;

namespace {

/// Let's assume that we only have vector.contract with the standard indexing
/// maps:
///    (m, n, k), A: (m, k), B: (k, n), C: (m, n).
/// We will represent this contract operation by a "@".
///
/// Given a matmul:
///
/// C = A @ B
///
/// This pass decides when to convert this matmul to:
///
/// A.T = transpose(A)
/// B.T = transpose(B)
/// C.T = B.T @ A.T
/// C = transpose(C.T)
///
/// This is useful when the "@" instruction that the hardware lowers to
/// has a specific layout (see VectorLayoutInterface for more information)
/// but the further uses of C expects a transposed layout to the produced
/// layout.
///
/// For example, for "@" lowering to AMDGPU MFMA instructions, the operands
/// have layout L and L.T and the result has the layout L.T .
/// So if you have a chain of matmuls:
///
/// C (L.T) = A (L) @ B (L.T)
/// E (L.T) = C (L.T)  @ D (L.T)
///            ^^^^^^^
///            Expected layout by instruction is L
///
/// To fix this, we can apply this transformation on the first matrix:
///
/// C.T (L.T) = B.T (L) @ A (L.T)
/// C   (L)   = transpose C.T (L.T)
/// E   (L.T) = C (L)  @ D (L.T)
///            ^^^^^
///            Layout matches the instruction!
///
/// Note that the mathematical formula
///   C = A @ B --> C.T = B.T @ A.T
/// is only defined on standard "@" function, it may be a different
/// transformation for other indexing maps.
struct AMDGPUPrepareForChainedMatmulPass final
    : impl::AMDGPUPrepareForChainedMatmulPassBase<
          AMDGPUPrepareForChainedMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  VectorContractOpInfo getOpInfo(vector::ContractionOp contract) const {
    auto maybeOpInfo = VectorContractOpInfo::inferFromIndexingMaps(
        contract.getIndexingMapsArray());
    assert(succeeded(maybeOpInfo) &&
           "contraction info for vector.contract should always be valid");
    return maybeOpInfo.value();
  }

  VectorValue swapDims(RewriterBase &rewriter, VectorValue val, int64_t dimA,
                       int64_t dimB) const {
    ArrayRef<int64_t> shape = val.getType().getShape();
    SmallVector<int64_t> perm(shape.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dimA], perm[dimB]);
    return rewriter.create<vector::TransposeOp>(val.getLoc(), val, perm);
  }

  AffineMap swapDimsInMap(AffineMap map, int64_t dimA, int64_t dimB) const {
    SmallVector<AffineExpr> results(map.getResults());
    std::swap(results[dimA], results[dimB]);
    return AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                          map.getContext());
  }

  /// Given a vector contract of the form
  /// %output = vector.contract %lhs, %rhs, %acc
  /// this function swaps the operands (%rhs, %lhs),
  /// transposes the accumulator and output and updates
  /// the indexing maps for the new contract op.
  ///
  /// Given a contract:
  ///
  ///   result = vector.contract lhs, rhs, acc
  ///
  /// transform it to
  ///
  ///   lhs.T = transpose(lhs)
  ///   rhs.T = transpose(rhs)
  ///   acc.T = transpose(acc)
  ///   result.T = vector.contract rhs.T, lhs.T, acc.T
  ///   result = transpose(result.T)
  ///
  /// This transformation holds for the "@" case we described above. For
  /// other indexing maps, we need to take into account transposed which are
  /// fused into the contract. `isOperandSwapInvariant` tells us when we can
  /// simply swap the operands without transposing them.
  void swapOperandsAndTranspose(RewriterBase &rewriter,
                                vector::ContractionOp contractOp) const {
    VectorContractOpInfo opInfo = getOpInfo(contractOp);
    auto [lhsM, rhsN] = opInfo.getOperandMNIndex();
    auto [lhsK, rhsK] = opInfo.getOperandKIndex();
    auto [accM, accN] = opInfo.getResultMNIndex();
    VectorValue lhs = contractOp.getLhs();
    VectorValue rhs = contractOp.getRhs();
    VectorValue acc = cast<VectorValue>(contractOp.getAcc());
    rewriter.setInsertionPoint(contractOp);

    SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
    AffineMap lhsMap = maps[0];
    AffineMap rhsMap = maps[1];
    AffineMap accMap = maps[2];

    acc = swapDims(rewriter, acc, accN, accM);
    accMap = swapDimsInMap(accMap, accN, accM);

    if (!isOperandSwapInvariant(contractOp)) {
      lhs = swapDims(rewriter, lhs, lhsK, lhsM);
      rhs = swapDims(rewriter, rhs, rhsK, rhsN);
      lhsMap = swapDimsInMap(lhsMap, lhsK, lhsM);
      rhsMap = swapDimsInMap(rhsMap, rhsK, rhsN);
    }

    auto swappedOp = rewriter.create<vector::ContractionOp>(
        contractOp.getLoc(), rhs, lhs, acc,
        rewriter.getAffineMapArrayAttr({rhsMap, lhsMap, accMap}),
        contractOp.getIteratorTypesAttr());
    swappedOp->setDiscardableAttrs(contractOp->getDiscardableAttrDictionary());

    acc = cast<VectorValue>(swappedOp.getResult());
    acc = swapDims(rewriter, acc, accN, accM);

    rewriter.replaceOp(contractOp, acc);
  }

  /// If one of the operands is transposed, while the other isn't, the
  /// transformation boils down to an operand swap and result transpose. This
  /// happens because transposing and swapping both operands, preserves the
  /// structure of the contraction. For example:
  ///
  /// def matmul_transpose_b(A, B):
  ///   B.T = transpose(B)
  ///   C = A @ B.T
  ///   return C
  ///
  /// def matmul_transpose_b_swapped(A, B):
  ///   A.T = transpose(A)
  ///   C.T = B @ A.T
  ///   C   = transpose(C.T)
  ///   return C
  ///
  /// matmul_transpose_b(B, A) = matmul_transpose_b_swapped(B, A).T
  ///
  /// For the sake of completeness, we also show that this does not hold
  /// when no operands are transposed, or both operands are transposed:
  ///
  /// def matmul(A, B):
  ///   C = A @ B
  ///   return C
  ///
  /// def matmul_swapped(A, B):
  ///  A.T = transpose(A)
  ///  B.T = transpose(B)
  ///  C.T = B.T @ A.T
  ///  C   = transpose(C.T)
  bool isOperandSwapInvariant(vector::ContractionOp contractOp) const {
    // Check if the innermost m, n, k dimensions are in the order:
    // lhs: (m, k), rhs: (n, k)
    VectorContractOpInfo opInfo = getOpInfo(contractOp);
    auto [lhsM, rhsN] = opInfo.getOperandMNIndex();
    auto [lhsK, rhsK] = opInfo.getOperandKIndex();
    bool isLhsTransposed = lhsM > lhsK;
    bool isRhsTransposed = rhsN < rhsK;
    return isLhsTransposed != isRhsTransposed;
  }

  /// Returns a vector.contract operation that this value was transitively
  /// produced from.
  ///
  /// A chained matmul is one where the lhs of the candidate matrix
  /// is a result of another matmul (a matmul lies in the backward slice of lhs
  /// of the first matmul).
  ///
  /// TODO: This definition of a chained matmul is crude. We should actually be
  /// checking if the layout of the result of the first matmul is transposed
  /// to that expected by the second matmul.
  FailureOr<vector::ContractionOp>
  getTransitiveMatmulParent(vector::ContractionOp contractOp) const {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    options.inclusive = true;
    getBackwardSlice(contractOp.getLhs(), &backwardSlice, options);
    vector::ContractionOp result;
    for (Operation *sliceOp : backwardSlice) {
      auto chainParent = dyn_cast<vector::ContractionOp>(sliceOp);
      if (!chainParent) {
        continue;
      }

      // For now, we only support transpose invariant matmuls. This is because
      // transposing the inputs may have a non-trivial cost which we need
      // to think about.
      // TODO: We should probably enable it always. Currently, this is
      // only useful in Flash Attention, where the first matmul is generally
      // a transpose.
      if (!isOperandSwapInvariant(chainParent)) {
        continue;
      }

      // If we have multiple matmul parents, we fail.
      if (result) {
        return failure();
      }

      result = chainParent;
    }

    if (result) {
      return result;
    }

    return failure();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<vector::ContractionOp> matmulCandidates;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      matmulCandidates.push_back(contractOp);
    });

    IRRewriter rewriter(funcOp.getContext());
    for (vector::ContractionOp candidate : matmulCandidates) {
      FailureOr<vector::ContractionOp> maybeChainedParent =
          getTransitiveMatmulParent(candidate);
      if (failed(maybeChainedParent)) {
        continue;
      }
      auto chainParent = maybeChainedParent.value();
      swapOperandsAndTranspose(rewriter, chainParent);

      // TODO: We should be only transposing the second matrix if the
      // result of the first matmul is used by the second matmul transitively.
      swapOperandsAndTranspose(rewriter, candidate);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
