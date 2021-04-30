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

#ifndef COMMON_GOSSIP_RPC_H_
#define COMMON_GOSSIP_RPC_H_

#include <string>

#include "brpc/channel.h"
#include "brpc/server.h"

#include "common/util/macro.h"

#include "common/gossip/proto/gossip.pb.h"

namespace common {
namespace gossip {
namespace rpc {

using RpcController = ::google::protobuf::RpcController;
using Closure = ::google::protobuf::Closure;
using Empty = ::google::protobuf::Empty;

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

  void Ping(RpcController* cntl, const Empty* req, Empty* _,
            Closure* done) override;
  void IndirectPing(RpcController* cntl, const PingReq* req, AckResp* resp,
                    Closure* done) override;
  void Suspect(RpcController* cntl, const SuspectMsg* req, Empty* _,
               Closure* done) override;
  void Sync(RpcController* cntl, const SyncMsg* req, SyncMsg* resp,
            Closure* done) override;
  void Dead(RpcController* cntl, const DeadMsg* req, Empty* resp,
            Closure* done) override;

 private:
  ServerImpl() = default;
  DISALLOW_COPY_AND_ASSIGN(ServerImpl);
};

}  // namespace rpc
}  // namespace gossip
}  // namespace common

#endif  // COMMON_GOSSIP_RPC_H_
