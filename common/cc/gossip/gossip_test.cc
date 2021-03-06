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

#include "common/cc/gossip/cluster.h"
#include "common/cc/util/time.h"

namespace common {
namespace gossip {

TEST(TestBroadcastQueue, TestSimple) {
  BroadcastQueue q(2);
  auto n1 = std::make_shared<Node>("1", 0, "0.0.0.0", 0, rpc::State::ALIVE, "");
  auto n2 = std::make_shared<Node>("2", 0, "0.0.0.0", 0, rpc::State::ALIVE, "");
  auto n3 = std::make_shared<Node>("3", 0, "0.0.0.0", 0, rpc::State::ALIVE, "");

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
  LOG(INFO) << q;
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
  auto n1 = std::make_shared<Node>("1", 0, "0.0.0.0", 0, rpc::State::ALIVE, "");

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

TEST(TestCluster, TestServer) {
  Cluster c(23331, 3, 100, 100, 100);
  c.Alive().Start();
  auto self = c.Nodes()[0];
  rpc::NodeMsg req;
  self->ToNodeMsg(&req);
  rpc::NodeMsg resp;
  rpc::Client::Send(self->addr(), req, &resp);
  std::cout << resp.ShortDebugString() << std::endl;
}

TEST(TestCluster, TestSimple) {
  Cluster c1(23331, 3, 100, 100, 100);
  c1.Alive("n1").Start();
  Cluster c2(23332, 3, 100, 100, 100);
  c2.Alive("n2").Start().Join("0.0.0.0", 23331);
  Cluster c3(23333, 3, 100, 100, 100);
  c3.Alive("n3").Start().Join("0.0.0.0", 23331);

  {
    util::SleepForMS(1 * 1000);
    std::cout << c1.ToString(true) << std::endl;
    std::cout << c2.ToString(true) << std::endl;
    std::cout << c3.ToString(true) << std::endl;

    c3.Stop();
    util::SleepForMS(1 * 1000);
    std::cout << c1.ToString(true) << std::endl;
    std::cout << c2.ToString(true) << std::endl;

    util::SleepForMS(1 * 60 * 1000);
    std::cout << c1.ToString(true) << std::endl;
    std::cout << c2.ToString(true) << std::endl;
  }

  {
    Cluster c3(23333, 3, 100, 100, 100);
    c3.Alive("n3").Start().Join("0.0.0.0", 23331);
    util::SleepForMS(1 * 1000);
    std::cout << c1.ToString(true) << std::endl;
    std::cout << c2.ToString(true) << std::endl;
    std::cout << c3.ToString(true) << std::endl;

    c3.Stop();
    util::SleepForMS(100);
    std::cout << c1.ToString(true) << std::endl;
    std::cout << c2.ToString(true) << std::endl;
  }

  {
    Cluster c3(23333, 3, 100, 100, 100);
    c3.Alive("n3").Start().Join("0.0.0.0", 23331);
    util::SleepForMS(1000);
    std::cout << c1.ToString(true) << std::endl;
    std::cout << c2.ToString(true) << std::endl;
    std::cout << c3.ToString(true) << std::endl;
  }
}

}  // namespace gossip
}  // namespace common
