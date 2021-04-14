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
    name = "gossip",
    srcs = ["gossip.cc"],
    hdrs = ["gossip.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gossip_cc_proto",
        "//common/net",
        "//common/util",
        "@sofa-pbrpc",
    ],
)

cc_test(
    name = "gossip_test",
    srcs = [
        "gossip_test.cc",
    ],
    deps = [
        ":gossip",
        "@com_google_googletest//:gtest_main",
    ],
)

cpplint()
