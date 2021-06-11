load("//bazel:latex.bzl", "latex_pdf")
load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary")

latex_pdf(
    name = "hw",
    srcs = [
        ":HW1/PIF_and_Tarry.tex",
        ":HW2/mutex.tex",
        ":HW4/Termination_and_Leader.tex",
        ":HW5/Arbitrary_Safra.tex",
    ],
    data = [
        ":HW2/p1.png",
        ":HW2/p3.png",
        ":HW2/p4.png",
        ":HW4/p1.png",
        ":HW4/p2.png",
        ":HW5/Bibliography.bib",
    ],
    deps = ["//latex:common"],
)

scala_binary(
    name = "PIF",
    srcs = [
        "prog1/Application.scala",
        "prog1/BroadcastConvergecast.scala",
        "prog1/PIFMain.scala",
    ],
    main_class = "prog1.PIFMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)

scala_binary(
    name = "CutVertices",
    srcs = [
        "prog2/CutVertices.scala",
        "prog2/CutVerticesMain.scala",
    ],
    main_class = "prog2.CutVerticesMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)

scala_binary(
    name = "RaymondMutex",
    srcs = [
        "prog3/MutexApplication.scala",
        "prog3/Raymond.scala",
        "prog3/RaymondMutexMain.scala",
        "prog3/SpanningTreeHelper.scala",
    ],
    main_class = "prog3.RaymondMutexMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)

scala_binary(
    name = "Safra",
    srcs = [
        "prog4/InformationDiffusion.scala",
        "prog4/Safra.scala",
        "prog4/SafraMain.scala",
        "prog4/SpanningTreeHelper.scala",
    ],
    main_class = "session3.SafraMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)
