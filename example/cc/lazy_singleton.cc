#include <iostream>

class A {
 public:
  static A& instance() {
    static A instance;
    return instance;
  }

  ~A() {
    dead_ = true;
    std::cout << "A destructor" << std::endl;
  }

  bool dead() { return dead_; }

 private:
  A() { std::cout << "A constructor "; }

  bool dead_ = false;
};

class B {
 public:
#ifdef WRONG
  B() = default;
#else
  B() { A::instance(); };
#endif
  ~B() { std::cout << "B destructor: " << A::instance().dead() << std::endl; }
  void work() { std::cout << "work: " << A::instance().dead() << std::endl; }
};

B b;
int main() { b.work(); }
