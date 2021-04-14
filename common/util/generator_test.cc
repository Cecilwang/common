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

#include <thread>  // NOLINT

#include "gtest/gtest.h"

#include "common/util/generator.h"

namespace common {
namespace util {

TEST(TestGenerator, TestReuse) {
  Generator<int> gen([](Generator<int>::Iterator* it) {
    for (int i = 0; i < 10; ++i) {
      it->Yield(i);
    }
  });
  int gt = 0;
  for (auto i : gen) {
    EXPECT_EQ(i, gt++);
  }
  gt = 0;
  for (auto i : gen) {
    EXPECT_EQ(i, gt++);
  }
}

TEST(TestGenerator, TestConcurrent) {
  Generator<int> gen([](Generator<int>::Iterator* it) {
    for (int i = 0; i < 10; ++i) {
      it->Yield(i);
    }
  });
  std::thread t1([&gen] {
    int gt = 0;
    for (auto i : gen) {
      EXPECT_EQ(i, gt++);
    }
  });
  std::thread t2([&gen] {
    int gt = 0;
    for (auto i : gen) {
      EXPECT_EQ(i, gt++);
    }
  });
  t1.join();
  t2.join();
}

}  // namespace util
}  // namespace common
