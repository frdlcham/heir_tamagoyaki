load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "MatmulKernelEGraph",
    srcs = ["MatmulKernelEGraph.cpp"],
    hdrs = ["MatmulKernelEGraph.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tamagoyaki//:MLIREquivalence",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "MatmulKernelEGraph",
)
