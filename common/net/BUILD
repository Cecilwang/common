package(default_visibility = ["//common:internal"])

load("//third_party/cpplint:cpplint.bzl", "cpplint")

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = ["net.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common/util",
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
