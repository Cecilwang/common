package_group(
    name = "internal",
    packages = [
        "//gossip/...",
    ],
)

package(default_visibility = ["//gossip:internal"])

load("//bazel/cpplint:cpplint.bzl", "cpplint")

cc_library(
    name = "gossip",
    srcs = ["gossip.cc"],
    hdrs = ["gossip.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//net",
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
