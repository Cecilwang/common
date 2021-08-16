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

#include "common/cc/util/thread.h"
#include "common/cc/util/time.h"

namespace common {
namespace util {

TEST(TestThread, TestDetach) {
  std::thread t([] { SleepForMS(10 * 1000); });
  t.detach();
}

TEST(TestThread, TestSimple) {
  int a = 0;
  {
    auto t = CreateThread([&a](Thread* p) { a = 1; });
    t->Run();
  }
  // XXX(sxwang): waiting for thread
  EXPECT_EQ(a, 1);
}

TEST(TestThread, TestIdle) {
  uint64_t ms = 1 * 1000;
  auto t = CreateThread([ms](Thread* p) {
    auto ts = NowInMS();
    p->WaitUntilStop();
    EXPECT_NEAR(NowInMS() - ts, ms, 100);
  });
  t->Run();
  SleepForMS(ms);
  t->Stop();
}

TEST(TestLoopThread, TestSimple) {
  uint64_t ms = 1 * 1000 + 100;
  uint64_t intvl_ms = 200;
  int count = 0;
  auto t = CreateLoopThread([&count] { ++count; }, intvl_ms);
  t->Run();
  SleepForMS(ms);
  t->Stop();
  EXPECT_EQ(count, ms / intvl_ms);
}

TEST(TestTimer, TestExcute) {
  uint64_t timeout_ms = 100;
  int i = 0;
  auto t = CreateTimer([&i] { ++i; }, timeout_ms);
  t->Run();
  SleepForMS(timeout_ms + 100);
  EXPECT_EQ(i, 1);
}

TEST(TestTimer, TestCancel) {
  uint64_t timeout_ms = 60 * 1000;
  int i = 0;

  {
    auto t = CreateTimer([&i] { ++i; }, timeout_ms);
    t->Run();
    SleepForMS(500);
  }
  EXPECT_EQ(i, 0);

  {
    auto t = CreateTimer([&i] { ++i; }, timeout_ms);
    t->Run();
    SleepForMS(500);
    t->Stop();
    EXPECT_EQ(i, 0);
  }
  EXPECT_EQ(i, 0);
}

TEST(TestTimer, TestChangeTimeout) {
  uint64_t timeout_ms = 60 * 1000;
  int i = 0;

  {
    auto t = CreateTimer([&i] { ++i; }, timeout_ms);
    t->Run();
    t->set_timeout_ms(100);
    SleepForMS(100 + 100);
    EXPECT_EQ(i, 1);
  }

  {
    auto t = CreateTimer([&i] { ++i; }, timeout_ms);
    t->Run();
    SleepForMS(100);
    t->set_timeout_ms(10 * 1000);
    SleepForMS(100);
    t->set_timeout_ms(1000);
    SleepForMS(1000);
    EXPECT_EQ(i, 2);
  }

  {
    auto t = CreateTimer([&i] { ++i; }, timeout_ms);
    t->Run();
    SleepForMS(500);
    t->set_timeout_ms(100);
    SleepForMS(10);
    EXPECT_EQ(i, 3);
  }
}

}  // namespace util
}  // namespace common
