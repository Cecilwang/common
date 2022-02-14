workspace(name = "common")

load("//:workspace.bzl", "common_deps")

common_deps()

load("//:workspace.bzl", "titech_deps")

titech_deps()

################### CC ###################

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

################### SCALA ###################

load("@io_bazel_rules_scala//:scala_config.bzl", "scala_config")

scala_config(scala_version = "2.13.5")

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
    name = "scalaneko_maven",
    artifacts = [
        "titech.c.coord:scalaneko_2.13:0.22.0",
    ],
    repositories = [
        "https://xdefago.github.io/ScalaNeko/sbt-repo/",
        "https://repo1.maven.org/maven2",
    ],
)

maven_install(
    name = "cg_maven",
    artifacts = [
        "org.openjfx:javafx-base:11.0.2",
        "org.openjfx:javafx-swing:11.0.2",
        "org.openjfx:javafx-controls:11.0.2",
        "org.openjfx:javafx-graphics:11.0.2",
        "org.openjfx:javafx-fxml:11.0.2",
        "org.bytedeco:opencv:4.5.3-1.5.6",
        "org.bytedeco:openblas:0.3.9-1.5.3",
        "org.bytedeco:javacv-platform:1.5.6",
        "org.jogamp.jogl:jogl-all-main:2.3.2",
        "gov.nist.math:jama:1.0.3",
    ],
    repositories = [
        "https://repo1.maven.org/maven2",
    ],
)

################### PYTHON ###################
load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
    name = "py_deps",
    python_interpreter = "python3",
    requirements = "//:requirements.txt",
)
