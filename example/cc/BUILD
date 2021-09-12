load("//third_party/cpplint:cpplint.bzl", "cpplint")

cc_binary(
    name = "shared_ptr",
    srcs = ["shared_ptr.cc"],
    linkopts = ["-lpthread"],
    deps = ["@gperftools//:tcmalloc_and_profiler"],
)

cc_binary(
    name = "move_constructor_and_assignment",
    srcs = ["move_constructor_and_assignment.cc"],
    deps = ["@gperftools//:tcmalloc_and_profiler"],
)

cc_binary(
    name = "locality",
    srcs = ["locality.cc"],
    deps = [
        "//common/cc/util",
        "@com_github_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cpplint()
