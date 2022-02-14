load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_jar")
load("//bazel:cmake_http_archive.bzl", "cmake_http_archive")
load("//third_party/brpc:brpc_deps.bzl", "brpc_deps")
load("//third_party/cpplint:cpplint_deps.bzl", "cpplint_deps")

def common_deps():
    #################### C/C++ ####################

    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.2.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.2.0.zip",
        sha256 = "e60cfd0a8426fa4f5fd2156e768493ca62b87d125cb35e94c44e79a3f0d8635f",
    )  # Used by cmake_http_archive, boost

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

    brpc_deps()

    cpplint_deps()

    #################### SCALA ####################

    http_archive(
        name = "io_bazel_rules_scala",
        sha256 = "b7fa29db72408a972e6b6685d1bc17465b3108b620cb56d9b1700cf6f70f624a",
        strip_prefix = "rules_scala-5df8033f752be64fbe2cedfd1bdbad56e2033b15",
        type = "zip",
        url = "https://github.com/bazelbuild/rules_scala/archive/5df8033f752be64fbe2cedfd1bdbad56e2033b15.zip",
    )

    #################### MAVEN ####################

    RULES_JVM_EXTERNAL_TAG = "4.2"
    RULES_JVM_EXTERNAL_SHA = "cd1a77b7b02e8e008439ca76fd34f5b07aecb8c752961f9640dea15e9e5ba1ca"

    http_archive(
        name = "rules_jvm_external",
        strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
        sha256 = RULES_JVM_EXTERNAL_SHA,
        url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
    )

    http_jar(
        name = "jogamp-fat",
        urls = [
            "https://jogamp.org/deployment/archive/rc/v2.4.0-rc-20200307/fat/jogamp-fat.jar",
        ],
    )

    #################### PYTHON ####################

    http_archive(
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.2.0/rules_python-0.2.0.tar.gz",
        sha256 = "778197e26c5fbeb07ac2a2c5ae405b30f6cb7ad1f5510ea6fdac03bded96cc6f",
    )

def titech_deps():
    http_file(
        name = "bert-base-uncased-vocab",
        downloaded_file_path = "bert-base-uncased-vocab.txt",
        urls = ["https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"],
    )
