/*
 * Copyright (c) 2019, Xavier Defago (Tokyo Institute of Technology).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package prog2

import neko._

class CutVertices(p: ProcessConfig)
  extends ActiveProtocol(p, "cut vertices")
    with StashedReceive
{
  import CutVertices.Msg

  private var r_i = 0         // the index of the round
  private var inf_i = Set(me) // the set of the process that have been received
  private var new_i = Set(me) // the set of the new messages
  private var com_with_i = neighbors // the set of unfinished neighbors
  private var routing_to_i = Map(me -> me) // received the key from the value at the first time
  // No need to update routing_to_i everytime when receiving the message
  // Because we only care about the process on the diameter of the elementary cycle
  private var dist_i       = Map(me -> 0) // the map of the distance

  /*
   * relationR is a map from PID to an equivalence class with respect to relation R.
   * Initially, each neighbor process is in a different equivalence class numbered
   * from 0 to the number of neighbors (minus one).
   * The private function transitiveAppend is used to state that two processes are
   * in the same class.
   * In the end, all processes are in the same equivalence class iff they are all in
   * class 0. If this is not the same (i.e., some process is in a class > 0), then
   * process `me` is a cut vertex because the channels corresponding to the neighbors
   * belong to different biconnected components.
   */
  private var relationR = allDifferent(neighbors)
  private def allDifferent(pids: Set[PID]): Map[PID,Int] = pids.zipWithIndex.map(pi => pi._1 -> pi._2).toMap
  private def isCutVertex: Boolean = relationR.values.exists(_ > 0)

  listenTo(classOf[Msg])

  def run() {
    while( new_i.nonEmpty ) {
      /* begin asynchronous round */

      r_i = r_i + 1 // next round

      SEND(Msg(me, com_with_i, r_i, new_i)) // Propagate new messages to all neighbors
      new_i = Set.empty[PID]                // Clean it

      for (x <- com_with_i) {
        ReceiveOrStash {
          case Msg(`x`,_,r,new_m) if r == r_i => // Only handle messages from this round
            if (new_m.isEmpty) {
              com_with_i = com_with_i - x // This process is done
            }
            for (p <- new_m) {
              if (!(inf_i(p)) && !(new_i(p))) { // It's the first time to receive p
                routing_to_i += (p->x)
                dist_i += (p->r_i)
              } else if (r_i == dist_i(p) || r_i == dist_i(p)+1) { // Is elementary cycle
                relationR = transitiveAppend(relationR, (routing_to_i(p), x))
              }
            }
            // Prepare the new message which will be sent in the next round
            // Moreover, it will only propagate each process once
            new_i = new_i ++ new_m -- inf_i
        }
      }

      inf_i = inf_i ++ new_i // Update received processes

      /* end asynchronous round */
    }

    // Notifing all neighbors that I'm done now.
    r_i = r_i + 1
    SEND(Msg(me, com_with_i, r_i, new_i))
    for (x <- com_with_i) ReceiveOrStash{case Msg(_,_,_,_) => /* do nothing */ }

    val isCutVertex = relationR.values.exists(_ > 0)

    if (isCutVertex) {
      println(s">>>>> ${me.name} IS CUT VERTEX")
    } else {
      println(s">>>>> ${me.name} not cut vertex")
    }
  }


  private def transitiveAppend(grouping: Map[PID,Int], related: (PID,PID)): Map[PID,Int] = {
    /*
     * grouping maps a PID to a group number. Two processes are in the same group if
     * they map to the same number. The function assigns the same grouping number to
     * the two related processes, taken as the minimum if one of them already has an
     * assignment.
     * All processes are in the same equivalence group, iff they all map to value 0.
     */
    var grpmap = grouping
    val (a,b) = related
    val default = grpmap.size
    val group_a = grpmap.getOrElse(a, default)
    val group_b = grpmap.getOrElse(b, default)
    grpmap = grpmap.updated(a, group_a)
    grpmap = grpmap.updated(b, group_b)
    if (group_a != group_b) {
      val toGroup = math.min(group_a, group_b)
      grpmap =
        grpmap.mapValues(g => if (g == group_a || g == group_b) toGroup else g).toMap
    }
    grpmap
  }
}

object CutVertices {
  case class Msg(from: PID, to: Set[PID], rnd: Int, received: Set[PID])
    extends MulticastMessage
}
