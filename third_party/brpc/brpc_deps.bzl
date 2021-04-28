load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def brpc_deps():
    http_archive(
        name = "com_github_google_leveldb",
        build_file = "//third_party/leveldb:leveldb.BUILD",
        strip_prefix = "leveldb-a53934a3ae1244679f812d998a4f16f2c7f309a6",
        url = "https://github.com/google/leveldb/archive/a53934a3ae1244679f812d998a4f16f2c7f309a6.tar.gz",
    )

    http_archive(
        name = "openssl",
        url = "https://github.com/openssl/openssl/archive/OpenSSL_1_1_0h.tar.gz",
        sha256 = "f56dd7d81ce8d3e395f83285bd700a1098ed5a4cb0a81ce9522e41e6db7e0389",
        strip_prefix = "openssl-OpenSSL_1_1_0h",
        build_file = "//third_party/openssl:openssl.BUILD",
    )

    git_repository(
        name = "brpc",
        remote = "https://github.com/apache/incubator-brpc",
        commit = "b3a948c9dca29632b3367529488e070852e31f11",
        patch_args = ["-p1"],
        patches = ["//third_party/brpc:b3a948c9dca29632b3367529488e070852e31f11.patch"],
        shallow_since = "1617642087 +0200",
    )
