package session0

import neko._

class Hello(p: ProcessConfig) extends ActiveProtocol(p, "hello")
{
  def run() {
    println(s"Process ${me.name} says: 'Hello Neko World!'")
  }
}

object HelloNeko
  extends Main(topology.Clique(2))(
    ProcessInitializer { p => new Hello(p) }
  )
