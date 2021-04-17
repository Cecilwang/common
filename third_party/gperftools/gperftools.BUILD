load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

configure_make(
    name = "tcmalloc_and_profiler",
    configure_options = [
        "--enable-shared=no",
        "--enable-frame-pointers",
        "--disable-libunwind",
    ],
    lib_source = ":all",
    out_static_libs = ["libtcmalloc_and_profiler.a"],
    visibility = ["//visibility:public"],
)
