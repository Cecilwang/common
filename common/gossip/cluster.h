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

#ifndef COMMON_GOSSIP_CLUSTER_H_
#define COMMON_GOSSIP_CLUSTER_H_

#include <atomic>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/net/net.h"
#include "common/util/macro.h"
#include "common/util/thread.h"

#include "common/gossip/proto/gossip.pb.h"

#include "common/gossip/node.h"
#include "common/gossip/rpc.h"

namespace common {
namespace gossip {

class BroadcastQueue {
 public:
  explicit BroadcastQueue(uint32_t n_transmit);

  void Push(Node::ConstPtr node);
  Node::ConstPtr Pop();

  size_t Size();

 private:
  struct Element {
    typedef std::shared_ptr<Element> Ptr;
    typedef const std::shared_ptr<Element>& ConstPtr;

    Element(Node::ConstPtr node, uint32_t n_transmit, uint32_t id);

    Node::ConstPtr node = nullptr;
    uint32_t n_transmit;  // The primary key
    uint32_t id;          // The secondary key
  };
  static bool ElementCmp(Element::ConstPtr a, Element::ConstPtr b);

  uint32_t n_transmit_;  // Transmission times

  std::mutex mutex_;
  std::condition_variable cv_;
  uint32_t id_ = 0;  // Indicate the order of enqueue
  std::set<Element::Ptr, decltype(ElementCmp)*> queue_;    // Balanced tree
  std::unordered_map<Node::Ptr, Element::Ptr> existence_;  // Better way?

  DISALLOW_COPY_AND_ASSIGN(BroadcastQueue);
};

class Cluster {
 public:
  explicit Cluster(uint16_t port = 2333, uint32_t n_transmit = 3,
                   uint64_t probe_inv_ms = 500,
                   uint64_t sync_inv_ms = 30 * 1000,
                   uint64_t gossip_inv_ms = 200);
  ~Cluster();

  Cluster& Alive(const std::string& name = "");

  Cluster& Start();
  bool StartServer();
  bool StartRoutine();

  Cluster& Stop();
  void StopServer();
  void StopRoutine();

  void Probe();
  void Sync();
  void Gossip();

  void SendProbe(Node::ConstPtr node);
  bool SendPing(Node::ConstPtr node, uint64_t timeout);
  bool SendSuspect(Node::ConstPtr node, uint64_t timeout);
  int SendIndirectPing(Node::ConstPtr broker, Node::ConstPtr target,
                       uint64_t timeout);

  void RecvAlive(rpc::AliveMsg* alive);
  void RecvSuspect(rpc::SuspectMsg* suspect);
  void RecvDead(rpc::DeadMsg* dead);

  void Refute(Node::Ptr node, uint32_t version);

  void Broadcast(Node::ConstPtr node);

  void ShuffleNodes();
  template <class F>
  std::unordered_set<Node::Ptr> GetRandomNodes(size_t k, F&& f);

  std::string ToString(bool verbose = false) const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Cluster& self);

 private:
  std::atomic<uint32_t> version_;
  std::string name_;
  Health health_;

  std::mutex nodes_mutex_;
  std::unordered_map<std::string, Node::Ptr> nodes_m_;
  std::vector<Node::Ptr> nodes_v_;
  // std::unordered_set<std::string> blacklist_;

  size_t probe_i_ = 0;
  uint64_t probe_inv_ms_ = 500;
  std::unique_ptr<util::Thread> probe_t_ = nullptr;
  uint64_t sync_inv_ms_ = 30 * 1000;
  std::unique_ptr<util::Thread> sync_t_ = nullptr;
  uint64_t gossip_inv_ms_ = 200;
  std::unique_ptr<util::Thread> gossip_t_ = nullptr;

  net::Address addr_;
  brpc::Server server_;

  BroadcastQueue queue_;

  DISALLOW_COPY_AND_ASSIGN(Cluster);
};

}  // namespace gossip
}  // namespace common

#endif  // COMMON_GOSSIP_CLUSTER_H_
