load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:cmake_http_archive.bzl", "cmake_http_archive")

def sofa_pbrpc_deps():
    http_archive(
        name = "boost",
        url = "https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.bz2/download",
        type = "tar.bz2",
        strip_prefix = "boost_1_61_0/",
        build_file = "//third_party/boost:boost.BUILD",
        sha256 = "a547bd06c2fd9a71ba1d169d9cf0339da7ebf4753849a8f7d6fdb8feee99b640",
    )

    cmake_http_archive(
        name = "snappy",
        url = "https://github.com/google/snappy/archive/refs/tags/1.1.8.zip",
        strip_prefix = "snappy-1.1.8",
        sha256 = "38b4aabf88eb480131ed45bfb89c19ca3e2a62daeb081bdf001cfb17ec4cd303",
    )

    git_repository(
        name = "sofa-pbrpc",
        remote = "https://github.com/baidu/sofa-pbrpc",
        commit = "68ef9412922649b760eab029d2a2ea1555d09b70",
        shallow_since = "1527146609 +0800",
        patch_args = ["-p1"],
        patches = ["//third_party/sofa-pbrpc:1.1.4.patch"],
    )
