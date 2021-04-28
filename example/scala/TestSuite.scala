import org.scalatest._

class TestSuite extends FlatSpec {
  "things" should "work" in {
    assert(HelloWord.message == "hello world")
  }
}
