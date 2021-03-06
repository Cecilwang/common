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

#ifndef COMMON_CC_GOSSIP_CLUSTER_H_
#define COMMON_CC_GOSSIP_CLUSTER_H_

#include <atomic>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "brpc/channel.h"
#include "brpc/server.h"

#include "common/cc/net/net.h"
#include "common/cc/util/macro.h"
#include "common/cc/util/thread.h"

#include "common/cc/gossip/proto/gossip.pb.h"

#include "common/cc/gossip/node.h"

namespace common {
namespace gossip {

class Cluster;

namespace rpc {

using RpcController = ::google::protobuf::RpcController;
using Closure = ::google::protobuf::Closure;

class Client {
 public:
  template <class REQ, class RESP>
  static bool Send(const net::Address& addr, const REQ& req, RESP* resp,
                   uint64_t timeout_ms = 500, int32_t n_retry = 1);

 private:
  template <class REQ, class RESP>
  static void Send(ServerAPI_Stub* stub, brpc::Controller* cntl, const REQ* req,
                   RESP* resp);

  Client() = delete;
};

class ServerImpl : public ServerAPI {
 public:
  explicit ServerImpl(Cluster* cluster);

  void Ping(RpcController* cntl, const NodeMsg* req, NodeMsg* resp,
            Closure* done) override;

  void Forward(RpcController* cntl, const ForwardMsg* req, NodeMsg* resp,
               Closure* done) override;

  void Sync(RpcController* cntl, const SyncMsg* req, SyncMsg* resp,
            Closure* done) override;

 private:
  Cluster* cluster_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(ServerImpl);
};

}  // namespace rpc

class BroadcastQueue {
 public:
  explicit BroadcastQueue(uint32_t n_transmit);

  void Push(Node::ConstPtrRef node);
  Node::Ptr Pop();

  size_t Size();

  std::string ToString();
  operator std::string();
  friend std::ostream& operator<<(std::ostream& os, BroadcastQueue& self);

 private:
  struct Element {
    typedef std::shared_ptr<Element> Ptr;
    typedef const std::shared_ptr<Element>& ConstPtrRef;

    Element(Node::ConstPtrRef node, uint32_t n_transmit, uint32_t id);

    Node::Ptr node = nullptr;
    uint32_t n_transmit;  // The primary key
    uint32_t id;          // The secondary key
  };
  static bool ElementCmp(Element::ConstPtrRef a, Element::ConstPtrRef b);

  uint32_t n_transmit_;  // Transmission times

  uint32_t id_ = 0;  // Indicate the order of enqueue
  std::set<Element::Ptr, decltype(ElementCmp)*> queue_;    // Balanced tree
  std::unordered_map<Node::Ptr, Element::Ptr> existence_;  // Better way?

  std::condition_variable cv_;
  std::mutex mutex_;

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

  void Join(const std::string& ip, uint16_t port);
  void Probe();
  void Probe(Node::ConstPtrRef node);
  void Sync();
  void Gossip();

  // Use Recv to dispatch the request and do not use others directly.
  void Recv(const rpc::NodeMsg* req, rpc::NodeMsg* resp);
  void RecvAlive(const rpc::NodeMsg* alive, rpc::NodeMsg* resp);
  void RecvSuspect(const rpc::NodeMsg* suspect, rpc::NodeMsg* resp);
  void RecvDead(const rpc::NodeMsg* dead, rpc::NodeMsg* resp);

  void RecvSync(const rpc::SyncMsg* req);

  void Refute(Node::Ptr node, uint32_t version, rpc::NodeMsg* resp);

  void Broadcast(Node::ConstPtrRef node);

  void GenNodeMsg(Node::ConstPtrRef src, rpc::NodeMsg* dst) const;
  rpc::NodeMsg GenNodeMsg(Node::ConstPtrRef node) const;
  rpc::NodeMsg GenNodeMsg(Node::ConstPtrRef node, rpc::State state) const;
  rpc::ForwardMsg GenForwardMsg(Node::ConstPtrRef node) const;
  rpc::ForwardMsg GenForwardMsg(Node::ConstPtrRef node, rpc::State state) const;
  void GenSyncMsg(rpc::SyncMsg* msg);
  rpc::SyncMsg GenSyncMsg();

  void ShuffleNodes();
  template <class F>
  std::vector<Node::Ptr> GetRandomNodes(size_t maxn, F&& f);

  std::vector<Node::Ptr> Nodes();

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

  BroadcastQueue queue_;

  net::Address addr_;
  brpc::Server server_;

  size_t probe_i_ = 0;
  uint64_t probe_inv_ms_ = 500;
  std::unique_ptr<util::Thread> probe_t_ = nullptr;
  uint64_t sync_inv_ms_ = 30 * 1000;
  std::unique_ptr<util::Thread> sync_t_ = nullptr;
  uint64_t gossip_inv_ms_ = 200;
  std::unique_ptr<util::Thread> gossip_t_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(Cluster);
};

}  // namespace gossip
}  // namespace common

#endif  // COMMON_CC_GOSSIP_CLUSTER_H_
