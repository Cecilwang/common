load("//bazel:latex.bzl", "latex_pdf")

latex_pdf(
    name = "hw",
    srcs = [
        ":Final/final.tex",
        #":HW1/h1.tex",
        #":HW2/h2.tex",
        #":HW3/h3.tex",
        #":HW4/h4.tex",
        #":HW5/h5.tex",
        #":HW6/h6.tex",
    ],
    data = [
        ":Final/lasso.png",
        ":Final/semi-log.png",
        ":Final/semi-log2.png",
        "//example/ml:1_binary_classification.py",
    ],
    deps = ["//latex:common"],
)
