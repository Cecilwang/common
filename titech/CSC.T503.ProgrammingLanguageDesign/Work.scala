trait Work {
  var count: Int = 0
  def add() = { count = count + 1 }
}

trait IntWork extends Work {
  def work(x: Int) = { add() }
}

trait FloatWork extends Work {
  def work(x: Float) = { add() }
}

class A extends IntWork {}

class B extends FloatWork {}

class C extends IntWork with FloatWork {}

object Main extends App {
  var c = new C()
  println(c.count)
  c.work(1)
  println(c.count)
  c.work(1.0f)
  println(c.count)
}
