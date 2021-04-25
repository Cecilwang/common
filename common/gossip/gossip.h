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
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "sofa/pbrpc/pbrpc.h"

#include "common/net/net.h"
#include "common/util/macro.h"

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
  Node(uint32_t version, const std::string& name, const std::string& ip,
       uint16_t port, rpc::State state, const std::string& metadata);
  explicit Node(const rpc::AliveMsg* alive);

  Node& operator=(const rpc::AliveMsg& alive);

  bool Conflict(const rpc::AliveMsg* alive) const;
  bool Reset(const rpc::AliveMsg* alive) const;

  uint32_t version() const;
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
  std::string ip_;
  uint16_t port_;
  rpc::State state_;
  std::string metadata_;

  DISALLOW_COPY_AND_ASSIGN(Node);
};

bool operator>(const Node& self, const rpc::AliveMsg& alive);
bool operator<=(const Node& self, const rpc::AliveMsg& alive);
bool operator==(const Node& self, const rpc::AliveMsg& alive);

class Cluster {
 public:
  explicit Cluster(uint16_t port = 2333, int32_t n_worker_ = 8);
  ~Cluster();

  Cluster& Start();
  Cluster& Stop();
  Cluster& Alive();

  void RecvAlive(rpc::AliveMsg* alive);

  std::string ToString(bool verbose = false) const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Cluster& self);

 private:
  std::atomic<uint32_t> version_;

  net::Address address_;

  std::mutex nodes_mutex_;
  std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;

  std::unordered_set<std::string> blacklist_;

  int32_t n_worker_;
  std::unique_ptr<sofa::pbrpc::RpcServer> server_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(Cluster);
};

}  // namespace gossip
}  // namespace common

#endif  // COMMON_GOSSIP_GOSSIP_H_
