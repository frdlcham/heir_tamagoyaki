#ifndef LIB_TRANSFORMS_MATMULKERNELEGRAPH_MATMULKERNELEGRAPH_H_
#define LIB_TRANSFORMS_MATMULKERNELEGRAPH_MATMULKERNELEGRAPH_H_

#include "EquivalenceDialect.h"  // from @tamagoyaki
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/MatmulKernelEGraph/MatmulKernelEGraph.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/MatmulKernelEGraph/MatmulKernelEGraph.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_MATMULKERNELEGRAPH_MATMULKERNELEGRAPH_H_
