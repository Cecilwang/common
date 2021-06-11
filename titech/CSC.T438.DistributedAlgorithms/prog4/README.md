I reuse the session3 to build this project in order to avoid build it from scratch. It uses a grid(2,2) topology by default now.

Considering the final report focus on Safra's algorithm, so the protocol of generating spanning tree is fake, and I also reuse SpanningTreeHelper.scala from project3. Unlike the original SpanningTreeHelper.scala, the spanningTree function will also return a map of children now.

I only added some private variables in class Safra, such as parent, children, expected, etc. Another modification is about the logical of receiving TokenCarrier according to my report. Finally, each process will broadcast Announce to all its neighbors, and use flags to avoid duplication. This is not stated in my report.
