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

#include "gtest/gtest.h"

#include "common/thread.h"
#include "common/time.h"

namespace common {

TEST(TestThread, TestSimple) {
  auto t = CreateThread([](Thread* p) { std::cout << "Running" << std::endl; });
  t->Run();
}

TEST(TestThread, TestIdle) {
  uint64_t ms = 1 * 1000;
  auto t = CreateThread([ms](Thread* p) {
    auto ts = NowInMS();
    p->WaitUntilStop();
    EXPECT_NEAR(NowInMS() - ts, ms, 100);
  });
  t->Run();
  t->Idle(ms);
  t->Stop();
}

TEST(TestLoopThread, TestSimple) {
  uint64_t ms = 1 * 1000 + 100;
  uint64_t interval_ms = 200;
  int count = 0;
  auto t = CreateLoopThread([&count] { ++count; }, interval_ms);
  t->Run();
  t->Idle(ms);
  t->Stop();
  EXPECT_EQ(count, ms / interval_ms);
}

}  // namespace common
