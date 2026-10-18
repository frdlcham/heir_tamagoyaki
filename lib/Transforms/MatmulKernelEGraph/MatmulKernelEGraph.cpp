#include "lib/Transforms/MatmulKernelEGraph/MatmulKernelEGraph.h"

#include <cstdint>
#include <numeric>

#include "EquivalenceDialect.h"  // from @tamagoyaki
#include "EquivalenceUtils.h"    // from @tamagoyaki
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "matmul-kernel-egraph"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_MATMULKERNELEGRAPH
#include "lib/Transforms/MatmulKernelEGraph/MatmulKernelEGraph.h.inc"

namespace {

using equivalence::ClassOp;
using equivalence::GraphOp;
using linalg::MatmulOp;
using tensor_ext::AssignLayoutOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

constexpr StringRef kLayoutAttrName =
    tensor_ext::TensorExtDialect::kLayoutAttrName;
// Default name of the attribute read by --equivalence-select-greedy.
constexpr StringRef kCostAttrName = "equivalence.cost";

LayoutAttr getLayout(Value value) {
  FailureOr<Attribute> attr =
      findAttributeAssociatedWith(value, kLayoutAttrName);
  if (failed(attr)) return nullptr;
  return dyn_cast<LayoutAttr>(attr.value());
}

bool isSameLayout(LayoutAttr lhs, LayoutAttr rhs) {
  return lhs == rhs ||
         isRelationEqual(lhs.getIntegerRelation(), rhs.getIntegerRelation());
}

LayoutAttr getBicyclicLayout(RankedTensorType type, int64_t ciphertextSize) {
  return LayoutAttr::getFromIntegerRelation(
      type.getContext(), getBicyclicLayoutRelation(type, ciphertextSize));
}

// The bicyclic packing requires coprime matrix dimensions (TODO(#2386): zero
// padding would lift this), and the bicyclic kernel operates on a single
// ciphertext per matrix.
bool supportsBicyclic(int64_t m, int64_t k, int64_t n, int64_t ciphertextSize) {
  return std::gcd(m, k) == 1 && std::gcd(k, n) == 1 && std::gcd(m, n) == 1 &&
         m * k <= ciphertextSize && k * n <= ciphertextSize &&
         m * n <= ciphertextSize;
}

// Returns a value equal to `value` but packed with `target` layout. Layout
// conversions feeding `value` are looked through, so that the new layout is
// derived from the original data rather than from a layout that was only
// chosen for another kernel. Plaintexts are re-packed directly.
FailureOr<Value> packWithLayout(OpBuilder& builder, Location loc, Value value,
                                LayoutAttr target) {
  LayoutAttr layout = getLayout(value);
  while (auto convertOp = value.getDefiningOp<ConvertLayoutOp>()) {
    auto fromLayout = dyn_cast<LayoutAttr>(convertOp.getFromLayout());
    if (!fromLayout) break;
    value = convertOp.getValue();
    layout = fromLayout;
  }

  if (auto assignOp = value.getDefiningOp<AssignLayoutOp>()) {
    auto assigned = dyn_cast<LayoutAttr>(assignOp.getLayout());
    if (assigned && isSameLayout(assigned, target)) return value;
    auto newAssignOp =
        AssignLayoutOp::create(builder, loc, assignOp.getValue(), target);
    newAssignOp->setAttr(kLayoutAttrName, target);
    return newAssignOp.getResult();
  }

  if (!layout) return failure();
  if (isSameLayout(layout, target)) return value;
  auto convertOp = ConvertLayoutOp::create(builder, loc, value, layout, target,
                                           builder.getDenseI64ArrayAttr({}));
  convertOp->setAttr(kLayoutAttrName, target);
  return convertOp.getResult();
}

struct MatmulKernelEGraph : impl::MatmulKernelEGraphBase<MatmulKernelEGraph> {
  using MatmulKernelEGraphBase::MatmulKernelEGraphBase;

  // Returns true if `op` is a ciphertext-plaintext matmul that was assigned
  // the (non-forced) MatmulDiagonal kernel by layout-propagation and that also
  // admits the MatmulBicyclic kernel.
  bool isCandidate(MatmulOp op, DataFlowSolver& solver) {
    if (op.hasUserDefinedMaps() || !op.hasPureTensorSemantics()) return false;

    auto kernel = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (!kernel || kernel.getName() != KernelName::MatmulDiagonal ||
        kernel.getForce())
      return false;
    if (!op->getAttrOfType<LayoutAttr>(kLayoutAttrName)) return false;

    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    if (!isSecret(lhs, &solver) || isSecret(rhs, &solver)) return false;

    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return false;

    return supportsBicyclic(lhsType.getDimSize(0), lhsType.getDimSize(1),
                            rhsType.getDimSize(1), ciphertextSize);
  }

  // Builds the bicyclic alternative to `op` at the builder's insertion point.
  // The returned value has the same layout as the result of `op`.
  FailureOr<Value> buildBicyclicMatmul(OpBuilder& builder, MatmulOp op) {
    MLIRContext* ctx = op.getContext();
    Location loc = op.getLoc();
    auto resultLayout = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
    auto resultType = cast<RankedTensorType>(op.getResultTensors()[0].getType());

    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      auto operandType = cast<RankedTensorType>(operand.getType());
      FailureOr<Value> packed = packWithLayout(
          builder, loc, operand, getBicyclicLayout(operandType, ciphertextSize));
      if (failed(packed)) return failure();
      newOperands.push_back(packed.value());
    }

    LayoutAttr bicyclicResultLayout =
        getBicyclicLayout(resultType, ciphertextSize);
    Operation* bicyclicOp = builder.clone(*op);
    bicyclicOp->setOperands(newOperands);
    // A cost annotation on `op` describes the diagonal kernel only.
    bicyclicOp->removeAttr(kCostAttrName);
    bicyclicOp->setAttr(
        secret::SecretDialect::kKernelAttrName,
        secret::KernelAttr::get(ctx, KernelName::MatmulBicyclic,
                                /*force=*/false));
    bicyclicOp->setAttr(kLayoutAttrName, bicyclicResultLayout);

    // Convert back so both members of the e-class share the original layout.
    auto convertOp = ConvertLayoutOp::create(
        builder, loc, bicyclicOp->getResult(0), bicyclicResultLayout,
        resultLayout, builder.getDenseI64ArrayAttr({}));
    convertOp->setAttr(kLayoutAttrName, resultLayout);
    return convertOp.getResult();
  }

  // Replaces `op` with an equivalence.graph that yields an e-class containing
  // `op` and its bicyclic alternative.
  LogicalResult wrapInNewGraph(MatmulOp op) {
    Value result = op.getResultTensors()[0];
    auto resultLayout = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);

    OpBuilder builder(op);
    auto graphOp =
        GraphOp::create(builder, op.getLoc(), TypeRange{result.getType()}, {});
    graphOp->setAttr(kLayoutAttrName, resultLayout);
    Block* body = builder.createBlock(&graphOp.getBody());

    // Layout ops that only exist to feed this kernel belong to the diagonal
    // alternative, so they are moved into the graph along with it.
    for (Value operand : op->getOperands()) {
      Operation* defOp = operand.getDefiningOp();
      if (isa_and_nonnull<AssignLayoutOp, ConvertLayoutOp>(defOp) &&
          defOp->hasOneUse() && defOp->getBlock() == op->getBlock())
        defOp->moveBefore(body, body->end());
    }
    op->moveBefore(body, body->end());

    builder.setInsertionPointToEnd(body);
    FailureOr<Value> bicyclic = buildBicyclicMatmul(builder, op);
    if (failed(bicyclic)) return failure();

    auto classOp = ClassOp::create(builder, op.getLoc(),
                                   TypeRange{result.getType()},
                                   ValueRange{result, bicyclic.value()},
                                   /*leader=*/Value{}, /*min_cost_index=*/nullptr);
    classOp->setAttr(kLayoutAttrName, resultLayout);
    equivalence::YieldOp::create(builder, op.getLoc(),
                                 ValueRange{classOp.getResult()});

    result.replaceUsesWithIf(graphOp.getResult(0), [&](OpOperand& use) {
      return !graphOp->isAncestor(use.getOwner());
    });
    return success();
  }

  // Adds the bicyclic alternative of `op` to the e-graph `op` is already in.
  LogicalResult addToExistingGraph(MatmulOp op) {
    Value result = op.getResultTensors()[0];
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    FailureOr<Value> bicyclic = buildBicyclicMatmul(builder, op);
    if (failed(bicyclic)) return failure();

    auto classOp = ClassOp::create(builder, op.getLoc(),
                                   TypeRange{result.getType()},
                                   ValueRange{result, bicyclic.value()},
                                   /*leader=*/Value{}, /*min_cost_index=*/nullptr);
    classOp->setAttr(kLayoutAttrName,
                     op->getAttrOfType<LayoutAttr>(kLayoutAttrName));
    // If `result` is already in an e-class, this nests the classes;
    // restoreClassInvariants merges them afterwards.
    result.replaceAllUsesExcept(classOp.getResult(), classOp);
    return success();
  }

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run secretness analysis.\n";
      signalPassFailure();
      return;
    }

    SmallVector<MatmulOp> candidates;
    getOperation()->walk([&](MatmulOp op) {
      if (isCandidate(op, solver)) candidates.push_back(op);
    });

    bool modifiedExistingGraph = false;
    for (MatmulOp op : candidates) {
      LLVM_DEBUG(llvm::dbgs() << "Adding bicyclic alternative for " << op
                              << "\n");
      auto parentGraph = dyn_cast<GraphOp>(op->getParentOp());
      LogicalResult result =
          parentGraph ? addToExistingGraph(op) : wrapInNewGraph(op);
      if (failed(result)) {
        op.emitError() << "failed to build bicyclic matmul alternative";
        signalPassFailure();
        return;
      }
      modifiedExistingGraph |= static_cast<bool>(parentGraph);
    }

    // The greedy driver behind this needs an IsolatedFromAbove root.
    if (modifiedExistingGraph &&
        failed(equivalence::restoreClassInvariants(getOperation()))) {
      getOperation()->emitError() << "failed to restore e-class invariants";
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
