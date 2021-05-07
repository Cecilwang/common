load("//bazel:latex.bzl", "latex_pdf")
load("@io_bazel_rules_scala//scala:scala.bzl", "scala_binary")

latex_pdf(
    name = "hw",
    srcs = [
        ":HW1/PIF_and_Tarry.tex",
        ":HW2/mutex.tex",
    ],
    data = [
        ":HW2/p1.png",
        ":HW2/p3.png",
        ":HW2/p4.png",
    ],
)

scala_binary(
    name = "HelloWorld",
    srcs = ["session0/a_hello_scala/HelloWorld.scala"],
    main_class = "session0.a_hello_scala.HelloScala",
    tags = ["scala"],
)

scala_binary(
    name = "HelloNeko",
    srcs = ["session0/b_hello_neko/HelloNeko.scala"],
    main_class = "session0.b_hello_neko.HelloNeko",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)

scala_binary(
    name = "Bounce",
    srcs = [
        "session1/a_bouncer/BounceMain.scala",
        "session1/a_bouncer/Bouncer.scala",
        "session1/a_bouncer/Greeting.scala",
        "session1/a_bouncer/Player.scala",
    ],
    main_class = "session1.a_bouncer.BounceMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)

scala_binary(
    name = "FIFO",
    srcs = [
        "session1/b_fifo/Application.scala",
        "session1/b_fifo/FIFOChannel.scala",
        "session1/b_fifo/FIFOTestMain.scala",
    ],
    #main_class = "session1.b_fifo.FIFORealMain",
    main_class = "session1.b_fifo.FIFOTestMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
)

scala_binary(
    name = "Learn",
    srcs = [
        "session1/c_learn/Application.scala",
        "session1/c_learn/LearnTopology.scala",
        "session1/c_learn/LearnTopologyClient.scala",
        "session1/c_learn/LearnTopologyMain.scala",
    ],
    main_class = "session1.c_learn.LearnTopologyMain",
    tags = ["scala"],
    deps = [
        "//third_party/scalaneko",
    ],
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
