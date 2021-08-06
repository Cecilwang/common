import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

abstract class Printable {
  public abstract void print();
}

class A extends Printable {
  public void print() {
    System.out.println("A");
  }
}

class B extends Printable {
  public void print() {
    System.out.println("B");
  }
}

public class Print {
  static void print(List<? extends Printable> objs) {
    for (Printable x : objs) {
      x.print();
    }
  }

  public static void main(String[] args) {
    ArrayList<A> as = new ArrayList();
    as.add(new A());
    print(as);
    ArrayList<B> bs = new ArrayList();
    bs.add(new B());
    print(bs);
  }
}
