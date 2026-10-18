// RUN: heir-opt --layout-propagation=ciphertext-size=32 --matmul-kernel-egraph=ciphertext-size=32 \
// RUN:   --equivalence-select-greedy=default-cost=1 --equivalence-extract=remove-graphs --canonicalize %s \
// RUN:   | FileCheck %s --implicit-check-not=equivalence --implicit-check-not=MatmulBicyclic

// With a uniform cost per op, the diagonal alternative needs one fewer layout
// conversion, so it is selected and the bicyclic alternative is removed.

// CHECK: #secret.kernel<name = "MatmulDiagonal"
// CHECK: func @ct_pt_matmul
// CHECK: secret.generic
// CHECK: linalg.matmul
// CHECK-NOT: linalg.matmul
// CHECK: secret.yield
func.func @ct_pt_matmul(%arg0: !secret.secret<tensor<3x2xf32>>, %arg1: tensor<2x5xf32>) -> !secret.secret<tensor<3x5xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x5xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<3x2xf32>>) {
  ^body(%input0: tensor<3x2xf32>):
    %1 = linalg.matmul ins(%input0, %arg1 : tensor<3x2xf32>, tensor<2x5xf32>) outs(%cst : tensor<3x5xf32>) -> tensor<3x5xf32>
    secret.yield %1 : tensor<3x5xf32>
  } -> !secret.secret<tensor<3x5xf32>>
  return %0 : !secret.secret<tensor<3x5xf32>>
}
