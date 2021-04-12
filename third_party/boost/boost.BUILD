package(default_visibility = ["//visibility:public"])

load("@rules_foreign_cc//foreign_cc:boost_build.bzl", "boost_build")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

boost_build(
    name = "boost",
    lib_source = ":all_srcs",
    out_headers_only = True,
    out_include_dir = ".",
)
