package_group(
    name = "internal",
    packages = [
        "//gossip/...",
    ],
)

package(default_visibility = ["//gossip:internal"])

load("//bazel/cpplint:cpplint.bzl", "cpplint")

#cc_library(
#    name = "time",
#    srcs = ["time.cc"],
#    hdrs = ["time.h"],
#    deps = ["//common"],
#)

cc_library(
    name = "gossip",
    hdrs = ["gossip.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
    ],
)

cc_test(
    name = "gossip_test",
    srcs = [
        "gossip_test.cc",
    ],
    deps = [
        ":gossip",
        "@gtest//:gtest_main",
    ],
)

cpplint()
