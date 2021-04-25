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
  Cluster c;
  std::cout << c << std::endl;
  std::cout << c.ToString(true) << std::endl;
  c.Start();
  std::cout << c.ToString(true) << std::endl;
  c.Start();
  c.Stop();
  std::cout << c.ToString(true) << std::endl;
  c.Stop();
}

TEST(TestCluster, TestDestructors) {
  Cluster c;
  c.Start();
}

TEST(TestCluster, TestPort) {
  Cluster c1;
  c1.Start();
  Cluster c2;
  c2.Start();
  Cluster c3(1111);
  c3.Start();
}

TEST(TestCluster, TestAlive) {
  Cluster cluster(2333);
  cluster.Alive().Start();
}

TEST(TestCluster, TestPing) {
  Cluster cluster(2333);
  cluster.Alive().Start();

  rpc::GossipClientImpl client;
  std::thread t([&client] {
    auto resp = client.Ping(net::Address("127.0.0.1", 2333), 1 * 1000);
    resp->PrintDebugString();
  });
  t.join();
}

}  // namespace gossip
}  // namespace common
