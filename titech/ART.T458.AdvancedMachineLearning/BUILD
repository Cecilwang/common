load("@py_deps//:requirements.bzl", "requirement")
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

py_binary(
    name = "lasso",
    srcs = ["Final1/lasso.py"],
    main = "lasso.py",
    deps = [
        requirement("numpy"),
        requirement("cvxpy"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "1",
    srcs = ["Final2/q1.py"],
    main = "q1.py",
    deps = [
        requirement("numpy"),
        requirement("torch"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "2",
    srcs = ["Final2/q2.py"],
    main = "q2.py",
    deps = [
        requirement("torch"),
    ],
)
