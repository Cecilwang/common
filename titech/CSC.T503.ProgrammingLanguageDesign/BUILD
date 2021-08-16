load("//bazel:latex.bzl", "latex_pdf")
load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary")

latex_pdf(
    name = "final",
    srcs = [":final.tex"],
    deps = ["//latex:common"],
)

java_binary(
    name = "Print",
    srcs = ["Print.java"],
    main_class = "Print",
)

scala_binary(
    name = "Work",
    srcs = [
        "Work.scala",
    ],
    main_class = "Main",
    tags = ["scala"],
)
