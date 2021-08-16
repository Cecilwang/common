package(default_visibility = ["//visibility:public"])

load("@py_deps//:requirements.bzl", "requirement")

py_library(
    name = "util",
    srcs = [
        "datasets.py",
        "metrics.py",
        "models.py",
        "problems.py",
    ],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        "//common/py/util",
    ],
)
