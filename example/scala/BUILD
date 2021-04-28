load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary", "scala_library", "scala_test")

scala_library(
    name = "HelloWorld",
    srcs = ["HelloWorld.scala"],
    tags = ["scala"],
)

scala_test(
    name = "TestSuite",
    srcs = ["TestSuite.scala"],
    tags = ["scala"],
    deps = [":HelloWorld"],
)

scala_binary(
    name = "Main",
    srcs = ["Main.scala"],
    main_class = "Main",
    tags = ["scala"],
    deps = [":HelloWorld"],
)
