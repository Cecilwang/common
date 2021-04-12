load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:cmake_http_archive.bzl", "cmake_http_archive")

def common_deps():
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.2.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.2.0.zip",
    )  # Used by cmake_http_archive, boost

    http_archive(
        name = "boost",
        url = "https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.bz2/download",
        type = "tar.bz2",
        strip_prefix = "boost_1_61_0/",
        build_file = "//third_party/boost:boost.BUILD",
        sha256 = "a547bd06c2fd9a71ba1d169d9cf0339da7ebf4753849a8f7d6fdb8feee99b640",
    )  # Used by sofa-pbrpc

    cmake_http_archive(
        name = "snappy",
        url = "https://github.com/google/snappy/archive/refs/tags/1.1.8.zip",
        strip_prefix = "snappy-1.1.8",
        sha256 = "38b4aabf88eb480131ed45bfb89c19ca3e2a62daeb081bdf001cfb17ec4cd303",
    )  # Used by sofa-pbrpc

    http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
    )

    http_archive(
        name = "com_github_google_glog",
        sha256 = "62efeb57ff70db9ea2129a16d0f908941e355d09d6d83c9f7b18557c0a7ab59e",
        strip_prefix = "glog-d516278b1cd33cd148e8989aec488b6049a4ca0b",
        urls = ["https://github.com/google/glog/archive/d516278b1cd33cd148e8989aec488b6049a4ca0b.zip"],
    )

    git_repository(
        name = "com_google_googletest",
        remote = "https://github.com/google/googletest",
        tag = "release-1.10.0",
    )

    git_repository(
        name = "com_google_protobuf",
        remote = "https://github.com/protocolbuffers/protobuf",
        tag = "v3.15.8",
    )

    git_repository(
        name = "sofa-pbrpc",
        remote = "https://github.com/baidu/sofa-pbrpc",
        tag = "v1.1.4",
        patch_args = ["-p1"],
        patches = ["//third_party/sofa-pbrpc:1.1.4.patch"],
    )
