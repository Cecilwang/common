package(default_visibility = ["//common:internal"])

load("@rules_proto//proto:defs.bzl", "proto_library")
load("//third_party/cpplint:cpplint.bzl", "cpplint")

proto_library(
    name = "gossip_proto",
    srcs = [":proto/gossip.proto"],
    deps = ["@com_google_protobuf//:empty_proto"],
)

cc_proto_library(
    name = "gossip_cc_proto",
    deps = [":gossip_proto"],
)

cc_library(
    name = "node",
    srcs = ["node.cc"],
    hdrs = ["node.h"],
    deps = [
        ":gossip_cc_proto",
        "//common/net",
        "//common/util",
    ],
)

cc_library(
    name = "cluster",
    srcs = ["cluster.cc"],
    hdrs = ["cluster.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gossip_cc_proto",
        ":node",
        "//common/net",
        "//common/util",
        "@brpc",
    ],
)

cc_test(
    name = "gossip_test",
    srcs = [
        "gossip_test.cc",
    ],
    deps = [
        ":cluster",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "gossip_bench",
    srcs = [
        "gossip_bench.cc",
    ],
    deps = [
        ":cluster",
        "@com_google_googletest//:gtest_main",
        "@gperftools//:tcmalloc_and_profiler",
    ],
)

cpplint()
