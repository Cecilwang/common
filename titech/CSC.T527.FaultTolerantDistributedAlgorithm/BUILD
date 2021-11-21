load("//bazel:latex.bzl", "latex_pdf")

latex_pdf(
    name = "hw1",
    srcs = [
        ":HW1.tex",
    ],
    deps = ["//latex:common"],
)

latex_pdf(
    name = "hw2",
    srcs = [
        ":HW2.tex",
    ],
    data = [
        ":hw2_1.png",
        ":hw2_2.png",
        ":hw2_3.png",
    ],
    deps = ["//latex:common"],
)

latex_pdf(
    name = "hw3",
    srcs = [
        ":HW3.tex",
    ],
    data = [
        ":hw3_1.png",
        ":hw3_2.png",
    ],
    deps = ["//latex:common"],
)

latex_pdf(
    name = "hw4",
    srcs = [
        ":HW4.tex",
    ],
    data = [
        ":hw4_1.png",
    ],
    deps = ["//latex:common"],
)
