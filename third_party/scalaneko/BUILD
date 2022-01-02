load("@io_bazel_rules_scala//scala:scala_import.bzl", "scala_import")

# It's pity that bazel_scala do not handle transitive dependencies.
# However, they pointed out that using direct dependencies can speed up compilation.
# Therefore, now we have to redefine the maven project with its dependencies again.

scala_import(
    name = "scalaneko",
    jars = [
        "@scalaneko_maven//:titech_c_coord_scalaneko_2_13",
        "@scalaneko_maven//:com_typesafe_scala_logging_scala_logging_2_13",
        "@scalaneko_maven//:ch_qos_logback_logback_classic",
        "@scalaneko_maven//:org_slf4j_slf4j_api",
    ],
    tags = ["scala"],
    visibility = ["//visibility:public"],
    runtime_deps = [
        "@scalaneko_maven//:ch_qos_logback_logback_core",
        "@scalaneko_maven//:com_typesafe_config",
        "@scalaneko_maven//:org_scala_graph_graph_core_2_13",
        # I'm not sure the following deps are necessary or not.
        # Maybe they will be added by scala_binary automated.
        # Just uncomment them on demand.
        #"@scalaneko_maven//:org_scala_lang_scala_library",
        #"@scalaneko_maven//:org_scala_lang_scala_reflect",
        #"@scalaneko_maven//:org_scalactic_scalactic_2_13",
        #"@scalaneko_maven//:org_scalafx_scalafx_2_13",
    ],
)
