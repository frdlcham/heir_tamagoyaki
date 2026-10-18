// RUN: heir-opt --layout-propagation=ciphertext-size=32 --matmul-kernel-egraph=ciphertext-size=32 %s | FileCheck %s --implicit-check-not=equivalence

// The result is 3x3, so the bicyclic layout (which needs coprime dimensions)
// does not apply.
// CHECK: func @non_coprime
// CHECK: linalg.matmul
func.func @non_coprime(%arg0: !secret.secret<tensor<3x2xf32>>, %arg1: tensor<2x3xf32>) -> !secret.secret<tensor<3x3xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<3x2xf32>>) {
  ^body(%input0: tensor<3x2xf32>):
    %1 = linalg.matmul ins(%input0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
    secret.yield %1 : tensor<3x3xf32>
  } -> !secret.secret<tensor<3x3xf32>>
  return %0 : !secret.secret<tensor<3x3xf32>>
}

// Each 7x5 matrix fills more than one 32-slot ciphertext.
// CHECK: func @too_large
// CHECK: linalg.matmul
func.func @too_large(%arg0: !secret.secret<tensor<7x2xf32>>, %arg1: tensor<2x5xf32>) -> !secret.secret<tensor<7x5xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<7x5xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<7x2xf32>>) {
  ^body(%input0: tensor<7x2xf32>):
    %1 = linalg.matmul ins(%input0, %arg1 : tensor<7x2xf32>, tensor<2x5xf32>) outs(%cst : tensor<7x5xf32>) -> tensor<7x5xf32>
    secret.yield %1 : tensor<7x5xf32>
  } -> !secret.secret<tensor<7x5xf32>>
  return %0 : !secret.secret<tensor<7x5xf32>>
}

// Ciphertext-ciphertext matmuls already use the bicyclic kernel.
// CHECK: func @ct_ct_matmul
// CHECK: linalg.matmul
func.func @ct_ct_matmul(%arg0: !secret.secret<tensor<3x2xf32>>, %arg1: !secret.secret<tensor<2x5xf32>>) -> !secret.secret<tensor<3x5xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x5xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<3x2xf32>>, %arg1: !secret.secret<tensor<2x5xf32>>) {
  ^body(%input0: tensor<3x2xf32>, %input1: tensor<2x5xf32>):
    %1 = linalg.matmul ins(%input0, %input1 : tensor<3x2xf32>, tensor<2x5xf32>) outs(%cst : tensor<3x5xf32>) -> tensor<3x5xf32>
    secret.yield %1 : tensor<3x5xf32>
  } -> !secret.secret<tensor<3x5xf32>>
  return %0 : !secret.secret<tensor<3x5xf32>>
}
