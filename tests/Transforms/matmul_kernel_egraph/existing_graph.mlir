// RUN: heir-opt --matmul-kernel-egraph=ciphertext-size=32 --mlir-print-local-scope %s | FileCheck %s
// RUN: heir-opt --matmul-kernel-egraph=ciphertext-size=32 --equivalence-select-greedy=default-cost=1 \
// RUN:   --equivalence-extract=remove-graphs --canonicalize --mlir-print-local-scope %s \
// RUN:   | FileCheck %s --check-prefix=EXTRACT --implicit-check-not=equivalence --implicit-check-not=MatmulDiagonal

#kernel = #secret.kernel<name = "MatmulDiagonal", force = false>
#out = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and (-i1 + slot) mod 8 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 4 and 0 <= slot <= 31 }">
#lhs = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-2i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1 and 0 <= slot <= 31 }">
#lhs_per_row = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and (i1 + slot) mod 2 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1 and 0 <= slot <= 31 }">
#rhs_diag = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 + i1 + ct) mod 2 = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 1 and 0 <= ct <= 1 and 0 <= slot <= 31 }">

// The matmul is already directly inside an e-graph, and its result is already
// in an e-class. The bicyclic alternative is merged into that e-class.

// CHECK: func @in_graph
// CHECK: %[[GRAPH:.*]] = equivalence.graph
// CHECK-NOT: equivalence.graph
// CHECK: %[[DIAG:.*]] = linalg.matmul {{.*}}"MatmulDiagonal"
// CHECK: %[[BI:.*]] = linalg.matmul
// CHECK-NOT: equivalence.cost
// CHECK-SAME: "MatmulBicyclic"
// CHECK: %[[BI_CONV:.*]] = tensor_ext.convert_layout %[[BI]]
// CHECK: %[[CLASS:.*]] = equivalence.class %[[DIAG]], %[[BI_CONV]]
// CHECK-NOT: equivalence.class
// CHECK: equivalence.yield %[[CLASS]]

// The diagonal kernel is made expensive, so extraction picks the bicyclic one.

// EXTRACT: func @in_graph
// EXTRACT: secret.generic
// EXTRACT: linalg.matmul {{.*}}"MatmulBicyclic"
// EXTRACT: tensor_ext.convert_layout
// EXTRACT: secret.yield
func.func @in_graph(%arg0: !secret.secret<tensor<3x2xf32>> {tensor_ext.layout = #lhs}, %arg1: tensor<2x5xf32>) -> (!secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #out}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x5xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<3x2xf32>> {tensor_ext.layout = #lhs}) {
  ^body(%input0: tensor<3x2xf32>):
    %1 = equivalence.graph -> (tensor<3x5xf32>) {
      %2 = tensor_ext.convert_layout %input0 {from_layout = #lhs, tensor_ext.layout = #lhs_per_row, to_layout = #lhs_per_row} : tensor<3x2xf32>
      %3 = tensor_ext.assign_layout %arg1 {layout = #rhs_diag, tensor_ext.layout = #rhs_diag} : tensor<2x5xf32>
      %4 = tensor_ext.assign_layout %cst {layout = #out, tensor_ext.layout = #out} : tensor<3x5xf32>
      %5 = linalg.matmul {equivalence.cost = #equivalence.cost<100>, secret.kernel = #kernel, tensor_ext.layout = #out} ins(%2, %3 : tensor<3x2xf32>, tensor<2x5xf32>) outs(%4 : tensor<3x5xf32>) -> tensor<3x5xf32>
      %6 = equivalence.class %5 : tensor<3x5xf32>
      equivalence.yield %6 : tensor<3x5xf32>
    }
    secret.yield %1 : tensor<3x5xf32>
  } -> (!secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<3x5xf32>>
}
