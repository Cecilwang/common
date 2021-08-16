load("@py_deps//:requirements.bzl", "requirement")

exports_files([
    "1_binary_classification.py",
])

py_binary(
    name = "1",
    srcs = ["1_binary_classification.py"],
    main = "1_binary_classification.py",
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "2",
    srcs = ["2_multiclass.py"],
    main = "2_multiclass.py",
    deps = [
        requirement("numpy"),
        requirement("jax"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "3",
    srcs = ["3_ggn.py"],
    main = "3_ggn.py",
    deps = [
        requirement("numpy"),
        requirement("jax"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "4",
    srcs = ["4_kfac.py"],
    main = "4_kfac.py",
    deps = [
        requirement("torch"),
        "//common/py/ml/kfac",
        "//common/py/ml/util",
    ],
)

py_binary(
    name = "5",
    srcs = ["5_loss_surface.py"],
    main = "5_loss_surface.py",
    deps = [
        requirement("torch"),
        requirement("numpy"),
        requirement("sklearn"),
        requirement("matplotlib"),
        "//common/py/ml/util",
    ],
)
