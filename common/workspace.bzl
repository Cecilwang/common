load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def common_workspace():
    git_repository(
        name = "gtest",
        remote = "https://github.com/google/googletest",
        tag = "release-1.10.0",
    )
