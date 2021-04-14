package(default_visibility = ["//common:internal"])

load("//third_party/cpplint:cpplint.bzl", "cpplint")

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
    name = "util",
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
    name = "util_test",
    srcs = [
        "generator_test.cc",
        "string_test.cc",
        "thread_test.cc",
        "type_test.cc",
    ],
    deps = [
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)

cpplint()
