/* Copyright 2021 cecilwang95@gmail.com
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <utility>

class A {
 public:
  A() { std::cout << "A default constructor" << std::endl; }
  ~A() { std::cout << "A default destructor" << std::endl; }

  explicit A(int a) : a_(a) {
    std::cout << "A copy constructor from int" << std::endl;
  }

  explicit A(const A& other) : a_(other.a_) {
    std::cout << "A copy constructor from A" << std::endl;
  }

  explicit A(A&& other) : a_(std::move(other.a_)) {
    std::cout << "A move constructor from A" << std::endl;
  }

  A& operator=(const A& other) {
    a_ = other.a_;
    std::cout << "A copy assignment" << std::endl;
    return *this;
  }

  A& operator=(A&& other) {
    a_ = std::move(other.a_);
    std::cout << "A move assignment" << std::endl;
    return *this;
  }

  int a_ = 0;
};

class B {
 public:
  B() { std::cout << "B default constructor" << std::endl; }
  ~B() { std::cout << "B default destructor" << std::endl; }

  explicit B(const A& a) : a_(a) {
    std::cout << "B copy constructor from A" << std::endl;
  }

  explicit B(A&& a) : a_(std::move(a)) {
    std::cout << "B move constructor from A" << std::endl;
  }

  explicit B(B&& other) : a_(std::move(other.a_)) {
    std::cout << "B move constructor from B" << std::endl;
  }

  A a_;
};

// A GetA() {
//  A a(1);
//  return a;
//}

int main() {
  std::cout << "A a1;" << std::endl;
  A a1;
  std::cout << "A a2(a1)" << std::endl;
  A a2(a1);
  std::cout << "A a3(std::move(a1))" << std::endl;
  A a3(std::move(a1));

  std::cout << "----------" << std::endl;

  std::cout << "B b1(A())" << std::endl;
  B b1((A()));

  std::cout << "----------" << std::endl;

  // std::cout << "A a4 = GetA()" << std::endl;
  // A a4 = GetA();

  // std::cout << "----------" << std::endl;
}
