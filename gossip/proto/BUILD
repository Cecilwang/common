package(default_visibility = ["//gossip:internal"])

load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "gossip_proto",
    srcs = ["gossip.proto"],
    deps = ["@com_google_protobuf//:empty_proto"],
)

cc_proto_library(
    name = "gossip_cc_proto",
    deps = [":gossip_proto"],
)
