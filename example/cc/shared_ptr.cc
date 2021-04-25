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

#include <assert.h>

#include <iostream>
#include <memory>
#include <utility>

static int ID;

void Priority() {
  std::cout << ++ID << ": -> is higher than ++" << std::endl;

  struct A {
    int a;
  };
  A* a = new A{0};
  ++a->a;
  assert(a->a == 1);
  delete a;
}

void UsedCount() {
  std::cout << ++ID
            << ": Always use const reference to pass shared_ptr. "
               "(https://stackoverflow.com/questions/3310737/"
               "should-we-pass-a-shared-ptr-by-reference-or-by-value)"
            << std::endl;
  std::cout << ++ID << ": reference will not increase use_count." << std::endl;

  std::shared_ptr<int> p = std::make_shared<int>(1);
  assert(p.use_count() == 1);
  {
    auto f = [](const std::shared_ptr<int>& p) -> const std::shared_ptr<int>& {
      // const reference will not increase use_count
      assert(p.use_count() == 1);
      return p;
    };
    const std::shared_ptr<int>& q = f(p);
    assert(p.use_count() == 1);
    assert(q.use_count() == 1);
    p = q;
    assert(p.use_count() == 1);
    assert(q.use_count() == 1);
  }
  assert(p.use_count() == 1);

  {
    auto f = [](const std::shared_ptr<int>& p) {
      assert(p.use_count() == 1);
      return p;
    };
    std::shared_ptr<int> q = f(p);
    assert(p.use_count() == 2);
    assert(q.use_count() == 2);
    p = q;
    assert(p.use_count() == 2);
    assert(q.use_count() == 2);
  }
  assert(p.use_count() == 1);

  {
    auto f = [](std::shared_ptr<int> p) {  // Donot pass shared_ptr by value;
      assert(p.use_count() == 2);
      return p;
    };
    std::shared_ptr<int> q = f(p);
    assert(p.use_count() == 2);
    assert(q.use_count() == 2);
    p = q;
    assert(p.use_count() == 2);
    assert(q.use_count() == 2);
  }
  assert(p.use_count() == 1);
}

int main() {
  Priority();
  UsedCount();
}
