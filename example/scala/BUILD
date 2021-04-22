load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary", "scala_library", "scala_test")

scala_library(
    name = "hello_world",
    srcs = ["hello_world.scala"],
    tags = ["scala"],
)

scala_test(
    name = "hello_world_test",
    srcs = ["hello_world_test.scala"],
    tags = ["scala"],
    deps = [":hello_world"],
)

scala_binary(
    name = "hello_world_main",
    srcs = ["hello_world_main.scala"],
    main_class = "Main",
    tags = ["scala"],
    deps = [":hello_world"],
)

scala_binary(
    name = "scalaneko_demo",
    srcs = ["scalaneko_demo.scala"],
    main_class = "session0.HelloNeko",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)
