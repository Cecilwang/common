package_group(
    name = "internal",
    packages = [
        "//net/...",
    ],
)

package(default_visibility = ["//net:internal"])

load("//bazel/cpplint:cpplint.bzl", "cpplint")

#cc_library(
#    name = "time",
#    srcs = ["time.cc"],
#    hdrs = ["time.h"],
#    deps = ["//common"],
#)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = ["net.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
    ],
)

cc_test(
    name = "net_test",
    srcs = [
        "net_test.cc",
    ],
    deps = [
        ":net",
        "@gtest//:gtest_main",
    ],
)

cpplint()
