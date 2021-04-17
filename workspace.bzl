load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:cmake_http_archive.bzl", "cmake_http_archive")

def common_deps():
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.2.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.2.0.zip",
        sha256 = "e60cfd0a8426fa4f5fd2156e768493ca62b87d125cb35e94c44e79a3f0d8635f",
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
        commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
        shallow_since = "1570114335 -0400",
        patch_args = ["-p0"],
        patches = ["//third_party/gtest:703bd9caab50b139428cea1aaff9974ebee5742e.patch"],
    )

    git_repository(
        name = "com_google_protobuf",
        remote = "https://github.com/protocolbuffers/protobuf",
        commit = "436bd7880e458532901c58f4d9d1ea23fa7edd52",
        shallow_since = "1617835118 -0700",
    )

    http_archive(
        name = "gperftools",
        build_file = "//third_party/gperftools:gperftools.BUILD",
        sha256 = "1ee8c8699a0eff6b6a203e59b43330536b22bbcbe6448f54c7091e5efb0763c9",
        strip_prefix = "gperftools-2.7",
        urls = [
            "https://github.com/gperftools/gperftools/releases/download/gperftools-2.7/gperftools-2.7.tar.gz",
        ],
    )

    git_repository(
        name = "sofa-pbrpc",
        remote = "https://github.com/baidu/sofa-pbrpc",
        commit = "68ef9412922649b760eab029d2a2ea1555d09b70",
        shallow_since = "1527146609 +0800",
        patch_args = ["-p1"],
        patches = ["//third_party/sofa-pbrpc:1.1.4.patch"],
    )
