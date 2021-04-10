package_group(
    name = "internal",
    packages = [
        "//common/...",
    ],
)

package(default_visibility = ["//common:internal"])

load("//bazel/cpplint:cpplint.bzl", "cpplint")

cc_library(
    name = "macro",
    hdrs = ["macro.h"],
)

cc_library(
    name = "type",
    hdrs = ["type.h"],
)

cc_library(
    name = "time",
    srcs = ["time.cc"],
    hdrs = ["time.h"],
    deps = [":type"],
)

cc_library(
    name = "string",
    srcs = ["string.cc"],
    hdrs = ["string.h"],
)

cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    hdrs = ["thread.h"],
    deps = [
        ":macro",
        ":time",
    ],
)

cc_library(
    name = "generator",
    hdrs = ["generator.h"],
    deps = [
        ":macro",
    ],
)

cc_library(
    name = "common",
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        ":string",
        ":thread",
        ":time",
        ":type",
    ],
)

cc_test(
    name = "common_test",
    srcs = [
        "generator_test.cc",
        "string_test.cc",
        "thread_test.cc",
        "type_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

cpplint()
