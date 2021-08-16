package(default_visibility = ["//visibility:public"])

load("@py_deps//:requirements.bzl", "requirement")

py_library(
    name = "kfac",
    srcs = [
        "__init__.py",
        "kfac.py",
    ],
    deps = [
        "//common/py/util",
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_test(
    name = "kfac_test",
    srcs = ["kfac_test.py"],
    deps = [
        "kfac",
        requirement("torch"),
    ],
)
