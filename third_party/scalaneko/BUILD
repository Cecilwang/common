load("@io_bazel_rules_scala//scala:scala_import.bzl", "scala_import")

scala_import(
    name = "scalaneko",
    jars = [
        #"@maven//:titech_c_coord_scalaneko",
        "@maven//:titech_c_coord_scalaneko_2_12",
        "@maven//:com_typesafe_scala_logging_scala_logging_2_12",
        "@maven//:ch_qos_logback_logback_classic",
        "@maven//:org_slf4j_slf4j_api",
    ],
    tags = ["scala"],
    visibility = ["//visibility:public"],
    runtime_deps = [
        "@maven//:ch_qos_logback_logback_core",
        "@maven//:com_typesafe_config",
        "@maven//:org_scala_graph_graph_core_2_12",
        "@maven//:org_scala_lang_scala_library",
        "@maven//:org_scala_lang_scala_reflect",
        "@maven//:org_scalactic_scalactic_2_12",
        "@maven//:org_scalafx_scalafx_2_12",
    ],
)
