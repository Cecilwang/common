package_group(
    name = "internal",
    packages = [
        "//net/...",
    ],
)

package(default_visibility = ["//net:internal"])

load("//third_party/cpplint:cpplint.bzl", "cpplint")

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = ["net.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "@com_github_google_glog//:glog",
    ],
)

cc_test(
    name = "net_test",
    srcs = [
        "net_test.cc",
    ],
    deps = [
        ":net",
        "@com_google_googletest//:gtest_main",
    ],
)

cpplint()
