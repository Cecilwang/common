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

#ifndef COMMON_GOSSIP_GOSSIP_H_
#define COMMON_GOSSIP_GOSSIP_H_

#include <atomic>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "sofa/pbrpc/pbrpc.h"

#include "common/net/net.h"
#include "common/util/macro.h"
#include "common/util/thread.h"

#include "common/gossip/proto/gossip.pb.h"

namespace common {
namespace gossip {
namespace rpc {

class GossipServerImpl : public GossipServerAPI {
 public:
  GossipServerImpl() = default;

  void Ping(::google::protobuf::RpcController* cntl, const PingReq* req,
            PingResp* resp, ::google::protobuf::Closure* done) override;
  void IndirectPing(::google::protobuf::RpcController* cntl,
                    const IndirectPingReq* req, PingResp* resp,
                    ::google::protobuf::Closure* done) override;
  void Sync(::google::protobuf::RpcController* cntl, const SyncMsg* req,
            SyncMsg* resp, ::google::protobuf::Closure* done) override;
  void Suspect(::google::protobuf::RpcController* cntl, const SuspectMsg* req,
               google::protobuf::Empty* resp,
               ::google::protobuf::Closure* done) override;
  void Dead(::google::protobuf::RpcController* cntl, const DeadMsg* req,
            google::protobuf::Empty* resp,
            ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(GossipServerImpl);
};

class GossipClientImpl {
 public:
  GossipClientImpl();

  std::unique_ptr<PingResp> Ping(const net::Address& address, uint64_t timeout);

 private:
  sofa::pbrpc::RpcClient client_;

  DISALLOW_COPY_AND_ASSIGN(GossipClientImpl);
};

}  // namespace rpc

class Node {
 public:
  typedef std::shared_ptr<Node> Ptr;
  typedef const std::shared_ptr<Node>& ConstPtr;

  Node(uint32_t version, const std::string& name, const std::string& ip,
       uint16_t port, rpc::State state, const std::string& metadata);
  explicit Node(const rpc::AliveMsg* alive);

  Node& operator=(const rpc::AliveMsg& alive);

  bool Conflict(const rpc::AliveMsg* alive) const;
  bool Reset(const rpc::AliveMsg* alive) const;

  void set_version(uint32_t version);

  uint32_t version() const;
  const std::string& name() const;
  const std::string& ip() const;
  uint16_t port() const;
  rpc::State state() const;
  const std::string& metadata() const;

  rpc::AliveMsg* ToAliveMsg() const;
  std::string ToString(bool verbose = false) const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Node& self);

 private:
  uint32_t version_;
  std::string name_;
  std::string ip_;
  uint16_t port_;
  rpc::State state_;
  std::string metadata_;

  DISALLOW_COPY_AND_ASSIGN(Node);
};

bool operator>(const Node& self, const rpc::AliveMsg& alive);
bool operator<=(const Node& self, const rpc::AliveMsg& alive);
bool operator==(const Node& self, const rpc::AliveMsg& alive);

class BroadcastQueue {
 public:
  explicit BroadcastQueue(uint32_t n_transmit);

  size_t Size();

  void Push(Node::ConstPtr node);
  Node::ConstPtr Pop();

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
  explicit Cluster(uint16_t port = 2333, int32_t n_worker_ = 8,
                   uint32_t n_transmit = 3);
  ~Cluster();

  Cluster& Alive();

  Cluster& Start();
  bool StartServer();
  bool StartRoutine();

  Cluster& Stop();
  void StopServer();
  void StopRoutine();

  void Probe();
  void Sync();
  void Gossip();

  void Broadcast(Node::ConstPtr node);

  void Refute(Node::Ptr node, uint32_t version);
  void RecvAlive(rpc::AliveMsg* alive);

  std::string ToString(bool verbose = false) const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Cluster& self);

 private:
  std::atomic<uint32_t> version_;

  int32_t n_worker_;
  net::Address address_;
  std::unique_ptr<sofa::pbrpc::RpcServer> server_ = nullptr;

  std::mutex nodes_mutex_;
  std::unordered_map<std::string, Node::Ptr> nodes_;
  std::unordered_set<std::string> blacklist_;
  BroadcastQueue queue_;

  uint64_t probe_intvl_ms_ = 500;
  uint64_t sync_intvl_ms_ = 30 * 1000;
  uint64_t gossip_intvl_ms_ = 200;
  std::unique_ptr<util::Thread> probe_thread_ = nullptr;
  std::unique_ptr<util::Thread> sync_thread_ = nullptr;
  std::unique_ptr<util::Thread> gossip_thread_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(Cluster);
};

}  // namespace gossip
}  // namespace common

#endif  // COMMON_GOSSIP_GOSSIP_H_
