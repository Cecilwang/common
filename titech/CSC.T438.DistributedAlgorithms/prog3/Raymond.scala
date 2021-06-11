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
package prog3

import neko._
import neko.util.MutexClient

class Raymond(p: ProcessConfig, initialParent: PID)
  extends ReactiveProtocol(p, "Raymond Mutex")
{
  import Raymond._

  private var interested_i = false
  private var parent_i   = initialParent
  private var queue_i    = Seq.empty[PID]

  private def useObject() {
    assert(parent_i == me)
    assert(interested_i)
    DELIVER(MutexClient.CanEnter)
  }

  def onSend = {
    /*
     * operation acquire_object()
     */
    case MutexClient.Request =>
      interested_i = true
      if (parent_i != me) {
        queue_i = queue_i :+ me
        if (queue_i.size == 1) {
          SEND(Request(me, parent_i))
        }
      } else {
        useObject
      }

    /*
     * operation release_object()
     */
    case MutexClient.Release =>
      interested_i = false
      if (!queue_i.isEmpty) {
        var k = queue_i.head
        queue_i = queue_i.drop(1)
        SEND(PrivObject(me, k))
        parent_i = k
        if (!queue_i.isEmpty) {
          SEND(Request(me, parent_i))
        }
      }
  }

  listenTo(classOf[Request])
  listenTo(classOf[PrivObject])

  def onReceive = {
    case Request(pk,_) =>
      if (parent_i == me) {
        if (interested_i) {
          queue_i = queue_i :+ pk
        } else {
          SEND(PrivObject(me, pk))
          parent_i = pk
        }
      } else {
        queue_i = queue_i :+ pk
        if (queue_i.size == 1) {
          SEND(Request(me, parent_i))
        }
      }

    case PrivObject(_,_) =>
      var k = queue_i.head
      queue_i = queue_i.drop(1)
      if (k == me) {
        parent_i = me
        useObject()
      } else {
        SEND(PrivObject(me, k))
        parent_i = k
        if (!queue_i.isEmpty) {
          SEND(Request(me, parent_i))
        }
      }
  }
}

object Raymond
{
  case class Request(from: PID, to: PID)    extends UnicastMessage
  case class PrivObject(from: PID, to: PID) extends UnicastMessage
}

