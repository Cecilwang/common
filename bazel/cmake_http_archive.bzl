load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "{name}",
    cache_entries = {{
        "CMAKE_C_FLAGS": "{c_flags}",
    }},
    lib_source = "//:all_srcs",
    out_static_libs = ["lib{name}.a"],
)
"""

def cmake_http_archive(c_flags = "-fPIC", **kwargs):
    if "build_file" in kwargs or "build_file_content" in kwargs:
        print("Do not specify build_file/build_file_content in cmake_http_archive")
    kwargs["build_file_content"] = _BUILD_FILE_CONTENT.format(name = kwargs["name"], c_flags = c_flags)
    http_archive(**kwargs)
