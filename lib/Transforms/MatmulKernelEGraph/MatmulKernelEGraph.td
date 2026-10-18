#ifndef LIB_TRANSFORMS_MATMULKERNELEGRAPH_MATMULKERNELEGRAPH_TD_
#define LIB_TRANSFORMS_MATMULKERNELEGRAPH_MATMULKERNELEGRAPH_TD_

include "mlir/Pass/PassBase.td"

def MatmulKernelEGraph : Pass<"matmul-kernel-egraph", "ModuleOp"> {
  let summary = "Express ciphertext-plaintext matmul kernel choices as an e-graph.";
  let description = [{
    This pass runs after `layout-propagation` and rewrites every
    `linalg.matmul` whose LHS is secret and whose RHS is a plaintext into an
    `equivalence.graph` (from Tamagoyaki) that yields a single
    `equivalence.class`. The class holds two equivalent implementations of the
    matmul:

    1. The original `MatmulDiagonal` matmul, with its per-row LHS, diagonal
       RHS and per-row result layouts as assigned by `layout-propagation`.
    2. A `MatmulBicyclic` matmul whose LHS, RHS and accumulator are packed
       with the bicyclic layout, followed by a `tensor_ext.convert_layout` from
       the bicyclic result layout to the original result layout.

    Both class members compute the same value in the same layout, so users of
    the original matmul are unaffected. To build the bicyclic alternative, the
    pass looks through `tensor_ext.convert_layout` ops that feed the matmul, so
    it converts from the original layout rather than from the per-row or
    diagonal one. It also re-packs plaintexts by creating a fresh
    `tensor_ext.assign_layout` on the cleartext value. Single-use layout ops
    that only exist to feed the diagonal kernel are moved into the graph, so
    each alternative's cost includes its own layout conversions.

    The bicyclic alternative is only added when all three matmul dimensions are
    pairwise coprime and each matrix fits in a single ciphertext. Otherwise the
    matmul is left untouched.

    If the matmul is already directly nested in an `equivalence.graph`, the
    alternatives are added to that graph instead of creating a new one.

    A kernel can then be chosen with the Tamagoyaki passes, e.g.
    `--equivalence-select-greedy --equivalence-extract=remove-graphs`.
  }];
  let options = [
    Option<"ciphertextSize", "ciphertext-size", "int64_t",
           /*default=*/"1024", "Number of slots in a ciphertext; must match the "
           "value used for `layout-propagation`.">,
  ];
  let dependentDialects = [
    "mlir::equivalence::EquivalenceDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::linalg::LinalgDialect",
  ];
}

#endif  // LIB_TRANSFORMS_MATMULKERNELEGRAPH_MATMULKERNELEGRAPH_TD_
