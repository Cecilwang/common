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
#include <thread>  // NOLINT

#include "gtest/gtest.h"

#include "common/gossip/gossip.h"
#include "common/util/time.h"

namespace common {
namespace gossip {

TEST(TestBroadcastQueue, TestSimple) {
  BroadcastQueue q(2);
  auto n1 = std::make_shared<Node>(0, "n1", "", 0, rpc::State::ALIVE, "");
  auto n2 = std::make_shared<Node>(0, "n2", "", 0, rpc::State::ALIVE, "");
  auto n3 = std::make_shared<Node>(0, "n3", "", 0, rpc::State::ALIVE, "");

  q.Push(n1);
  q.Push(n2);

  EXPECT_EQ(q.Pop(), n2);
  EXPECT_EQ(q.Size(), 2);
  EXPECT_EQ(q.Pop(), n1);
  EXPECT_EQ(q.Size(), 2);

  q.Push(n3);
  EXPECT_EQ(q.Pop(), n3);
  EXPECT_EQ(q.Size(), 3);
  EXPECT_EQ(q.Pop(), n3);
  EXPECT_EQ(q.Size(), 2);

  q.Push(n1);
  EXPECT_EQ(q.Pop(), n1);
  EXPECT_EQ(q.Size(), 2);
  EXPECT_EQ(q.Pop(), n1);
  EXPECT_EQ(q.Size(), 1);
  EXPECT_EQ(q.Pop(), n2);
  EXPECT_EQ(q.Size(), 0);
}

TEST(TestBroadcastQueue, TestConcurrent) {
  uint64_t ms = 1 * 1000;

  BroadcastQueue q(1);
  auto n1 = std::make_shared<Node>(0, "n1", "", 0, rpc::State::ALIVE, "");

  std::thread t1([&]() {
    auto ts = util::NowInMS();
    EXPECT_EQ(q.Pop(), n1);
    EXPECT_GE(util::NowInMS() - ts, ms - 100);
  });

  std::thread t2([&]() {
    util::SleepForMS(ms);
    q.Push(n1);
  });

  t1.join();
  t2.join();
}

TEST(TestCluster, TestLog) {
  Cluster c(2333, 8, 3, 100, 100, 100);
  EXPECT_EQ(c.ToString(), "Cluster(0.0.0.0:2333)");
  EXPECT_EQ(c.ToString(true),
            "Cluster(version: 0 state: down, address: 0.0.0.0:2333)");
  c.Start();
  EXPECT_EQ(c.ToString(true),
            "Cluster(version: 0 state: up, address: 0.0.0.0:2333)");
  c.Start();
  c.Stop();
  c.Stop();
}

TEST(TestCluster, TestPort) {
  Cluster c1(2333, 8, 3, 100, 100, 100);
  c1.Start();
  Cluster c2(2333, 8, 3, 100, 100, 100);
  c2.Start();
  Cluster c3(1111, 8, 3, 100, 100, 100);
  c3.Start();
}

TEST(TestCluster, TestPing) {
  Cluster c(2333, 8, 3, 100, 100, 100);
  c.Alive().Start();
}

}  // namespace gossip
}  // namespace common
