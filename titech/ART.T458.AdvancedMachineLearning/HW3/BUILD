load("@py_deps//:requirements.bzl", "requirement")

exports_files([
    "h3.tex",
    "h3.py",
])

py_binary(
    name = "h3",
    srcs = ["h3.py"],
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
    ],
)
