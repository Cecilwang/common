load("//bazel:latex.bzl", "latex_pdf")
load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary")

latex_pdf(
    name = "hw",
    srcs = [
        #":HW1/h1.tex",
        #":HW2/h2.tex",
        #":HW3/h3.tex",
        #":HW4/h4.tex",
        #":HW5/h5.tex",
        ":HW6/h6.tex",
    ],
    data = [
        "//example/ml:1_binary_classification.py",
    ],
    deps = ["//latex:common"],
)
