// RUN: heir-opt --layout-propagation=ciphertext-size=32 --matmul-kernel-egraph=ciphertext-size=32 %s | FileCheck %s

// CHECK-DAG: #[[DIAG_KERNEL:kernel[0-9]*]] = #secret.kernel<name = "MatmulDiagonal", force = false>
// CHECK-DAG: #[[BI_KERNEL:kernel[0-9]*]] = #secret.kernel<name = "MatmulBicyclic", force = false>
// Bicyclic layouts of the 3x2 lhs, the 2x5 rhs and the 3x5 result.
// CHECK-DAG: #[[BI_LHS:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (2i0 + 3i1 + slot) mod 6 = 0
// CHECK-DAG: #[[BI_RHS:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (5i0 + 4i1 + slot) mod 10 = 0
// CHECK-DAG: #[[BI_OUT:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (5i0 - 6i1 + slot) mod 15 = 0

// CHECK: func @ct_pt_matmul
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<2x5xf32>
// CHECK: %[[CST:.*]] = arith.constant
// CHECK: secret.generic
// CHECK-NEXT: ^body(%[[INPUT:.*]]: tensor<3x2xf32>):
// CHECK-NEXT: %[[GRAPH:.*]] = equivalence.graph -> (tensor<3x5xf32>) {

// The diagonal alternative, with the layout conversions that feed it.
// CHECK: %[[DIAG:.*]] = linalg.matmul {secret.kernel = #[[DIAG_KERNEL]], tensor_ext.layout = #[[OUT:layout[0-9]*]]}

// The bicyclic alternative converts from the original layouts.
// CHECK-DAG: %[[LHS:.*]] = tensor_ext.convert_layout %[[INPUT]] {{{.*}}to_layout = #[[BI_LHS]]}
// CHECK-DAG: %[[RHS:.*]] = tensor_ext.assign_layout %[[ARG1]] {layout = #[[BI_RHS]]
// CHECK-DAG: %[[ACC:.*]] = tensor_ext.assign_layout %[[CST]] {layout = #[[BI_OUT]]
// CHECK: %[[BI:.*]] = linalg.matmul {secret.kernel = #[[BI_KERNEL]], tensor_ext.layout = #[[BI_OUT]]} ins(%[[LHS]], %[[RHS]] : {{.*}}) outs(%[[ACC]] : {{.*}})
// CHECK: %[[BI_CONV:.*]] = tensor_ext.convert_layout %[[BI]] {from_layout = #[[BI_OUT]], tensor_ext.layout = #[[OUT]], to_layout = #[[OUT]]}

// CHECK: %[[CLASS:.*]] = equivalence.class %[[DIAG]], %[[BI_CONV]] {tensor_ext.layout = #[[OUT]]} : tensor<3x5xf32>
// CHECK: equivalence.yield %[[CLASS]] : tensor<3x5xf32>
// CHECK-NEXT: } {tensor_ext.layout = #[[OUT]]}
// CHECK-NEXT: secret.yield %[[GRAPH]]
func.func @ct_pt_matmul(%arg0: !secret.secret<tensor<3x2xf32>>, %arg1: tensor<2x5xf32>) -> !secret.secret<tensor<3x5xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x5xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<3x2xf32>>) {
  ^body(%input0: tensor<3x2xf32>):
    %1 = linalg.matmul ins(%input0, %arg1 : tensor<3x2xf32>, tensor<2x5xf32>) outs(%cst : tensor<3x5xf32>) -> tensor<3x5xf32>
    secret.yield %1 : tensor<3x5xf32>
  } -> !secret.secret<tensor<3x5xf32>>
  return %0 : !secret.secret<tensor<3x5xf32>>
}
