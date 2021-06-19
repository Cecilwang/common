load("//bazel:latex.bzl", "latex_pdf")
load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary")

latex_pdf(
    name = "hw",
    srcs = [
        ":HW1/h1.tex",
        ":HW2/h2.tex",
        "//titech/ART.T458.AdvancedMachineLearning/HW3:h3.tex",
    ],
    data = [
        "//titech/ART.T458.AdvancedMachineLearning/HW3:h3.py",
    ],
    deps = ["//latex:common"],
)
