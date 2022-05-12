package(default_visibility = ["//visibility:public"])

load("@py_deps//:requirements.bzl", "requirement")

py_library(
    name = "util",
    srcs = [
        "dist.py",
        "ifvp.py",
        "metrics.py",
    ],
    deps = [
        requirement("torch"),
        requirement("wandb"),
    ],
)
