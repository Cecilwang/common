load("//bazel:latex.bzl", "latex_pdf")

latex_pdf(
    name = "hw",
    srcs = [
        ":Final1/final1.tex",
        ":Final2/final2.tex",
        ":HW1/h1.tex",
        ":HW2/h2.tex",
        ":HW3/h3.tex",
        ":HW4/h4.tex",
        ":HW5/h5.tex",
        ":HW6/h6.tex",
    ],
    data = [
        ":Final1/lasso.png",
        ":Final1/semi-log.png",
        ":Final1/semi-log2.png",
        ":Final2/q1.png",
        ":Final2/sigmoid.png",
        "//example/ml:1_binary_classification.py",
    ],
    deps = ["//latex:common"],
)
