load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def cpplint_deps():
    http_archive(
        name = "cpplint",
        build_file = "//third_party/cpplint:cpplint.BUILD",
        sha256 = "05f879aab5a04307e916e32afb547567d8a44149ddc2f91bf846ce2650ce6d7d",
        strip_prefix = "cpplint-1.4.4",
        urls = [
            "https://github.com/cpplint/cpplint/archive/1.4.4.tar.gz",
        ],
    )
