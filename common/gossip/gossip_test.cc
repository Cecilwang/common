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

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "common/gossip/gossip.h"

namespace common {
namespace gossip {

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

TEST(TestCluster, TestPing) {
  Cluster cluster(2333);
  cluster.Start();

  GossipClientImpl client;
  std::thread t([&client] {
    auto resp = client.Ping(net::Address("127.0.0.1", 2333), 1 * 1000);
    resp->PrintDebugString();
  });
  t.join();
}

}  // namespace gossip
}  // namespace common
