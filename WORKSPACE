workspace(name = "common")

load("//:workspace.bzl", "common_deps")

common_deps()

################### CC ###################

load("//third_party/cpplint:cpplint_deps.bzl", "cpplint_deps")

cpplint_deps()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

################### SCALA ###################

load("@io_bazel_rules_scala//:scala_config.bzl", "scala_config")

scala_config(scala_version = "2.12.11")

load("@io_bazel_rules_scala//scala:scala.bzl", "scala_repositories")

scala_repositories()

load("@io_bazel_rules_scala//scala:toolchains.bzl", "scala_register_toolchains")

scala_register_toolchains()

load("@io_bazel_rules_scala//testing:scalatest.bzl", "scalatest_repositories", "scalatest_toolchain")

scalatest_repositories()

scalatest_toolchain()

################### MAVEN ###################

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "titech.c.coord:scalaneko_2.12:0.22.0",
    ],
    repositories = [
        "https://xdefago.github.io/ScalaNeko/sbt-repo/",
        "https://repo1.maven.org/maven2",
    ],
)
