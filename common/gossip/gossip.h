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
#include <vector>

#include "brpc/channel.h"
#include "brpc/server.h"

#include "common/net/net.h"
#include "common/util/macro.h"
#include "common/util/thread.h"

#include "common/gossip/proto/gossip.pb.h"

namespace common {
namespace gossip {
namespace rpc {

class Client {
 public:
  template <class REQ, class RESP>
  static void Send(ServerAPI_Stub* stub, brpc::Controller* cntl, const REQ* req,
                   RESP* resp);

  template <class REQ, class RESP>
  static bool Send(const std::string& ip, uint16_t port, const REQ& req,
                   RESP* resp, uint64_t timeout_ms = 500, int32_t n_retry = 3);

 private:
  Client() = delete;
};

class ServerImpl : public ServerAPI {
 public:
  static ServerImpl& Get();

  void Ping(::google::protobuf::RpcController* cntl,
            const ::google::protobuf::Empty* req, ::google::protobuf::Empty* _,
            ::google::protobuf::Closure* done) override;
  void IndirectPing(::google::protobuf::RpcController* cntl, const PingReq* req,
                    AckResp* resp, ::google::protobuf::Closure* done) override;
  void Suspect(::google::protobuf::RpcController* cntl, const SuspectReq* req,
               google::protobuf::Empty* _,
               ::google::protobuf::Closure* done) override;
  void Sync(::google::protobuf::RpcController* cntl, const SyncMsg* req,
            SyncMsg* resp, ::google::protobuf::Closure* done) override;
  void Dead(::google::protobuf::RpcController* cntl, const DeadMsg* req,
            google::protobuf::Empty* resp,
            ::google::protobuf::Closure* done) override;

 private:
  ServerImpl() = default;
  DISALLOW_COPY_AND_ASSIGN(ServerImpl);
};

}  // namespace rpc

struct Health {
  int score();

  Health& operator+=(int val);
  int operator*(int val);

 private:
  std::mutex mutex_;
  int score_ = 1;
  int upper_ = 8;
};

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

  uint32_t version() const;
  void set_version(uint32_t version);
  const std::string& name() const;
  const std::string& ip() const;
  uint16_t port() const;
  rpc::State state() const;
  const std::string& metadata() const;

  std::string ToString(bool verbose = false) const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Node& self);

 private:
  uint32_t version_;
  std::string name_;
  net::Address addr_;
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
  void ShuffleNodes();
  template <class F>
  std::unordered_set<Node::Ptr> GetRandomNodes(size_t k, F&& f);

  void RecvAlive(rpc::AliveMsg* alive);
  void RecvSuspect(rpc::SuspectReq* suspect);

  void Refute(Node::Ptr node, uint32_t version);

  void Broadcast(Node::ConstPtr node);

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

#endif  // COMMON_GOSSIP_GOSSIP_H_
