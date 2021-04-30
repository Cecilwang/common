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

#include "common/gossip/rpc.h"

#include "glog/logging.h"

namespace common {
namespace gossip {
namespace rpc {

#define DefineSend(FUNC, REQ, RESP)                               \
  template <>                                                     \
  void Client::Send(ServerAPI_Stub* stub, brpc::Controller* cntl, \
                    const REQ* req, RESP* resp) {                 \
    stub->FUNC(cntl, req, resp, nullptr);                         \
  }

DefineSend(Ping, Empty, Empty);
DefineSend(IndirectPing, PingReq, AckResp);
DefineSend(Suspect, SuspectMsg, Empty);

#undef DefineSend

template <class REQ, class RESP>
bool Client::Send(const std::string& ip, uint16_t port, const REQ& req,
                  RESP* resp, uint64_t timeout_ms, int32_t n_retry) {
  brpc::ChannelOptions options;
  options.timeout_ms = timeout_ms;
  options.max_retry = n_retry;

  brpc::Channel channel;
  if (channel.Init(ip.c_str(), port, &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel";
    return false;
  }

  brpc::Controller cntl;
  // cntl.set_log_id(log_id ++);

  ServerAPI_Stub stub(&channel);

  Send(&stub, &cntl, &req, resp);

  if (cntl.Failed()) {
    LOG(ERROR) << "Failed to send to " << cntl.remote_side() << ": errcode["
               << cntl.ErrorCode() << "], errMessage[" << cntl.ErrorText()
               << "]";
    return false;
  }
  return true;
}

template bool Client::Send(const std::string& ip, uint16_t port,
                           const Empty& req, Empty* resp, uint64_t timeout_ms,
                           int32_t n_retry);

template bool Client::Send(const std::string& ip, uint16_t port,
                           const PingReq& req, AckResp* resp,
                           uint64_t timeout_ms, int32_t n_retry);

template bool Client::Send(const std::string& ip, uint16_t port,
                           const SuspectMsg& req, Empty* resp,
                           uint64_t timeout_ms, int32_t n_retry);

//------------------------------------------------------------------------------

ServerImpl& ServerImpl::Get() {
  static ServerImpl instance;
  return instance;
}

void ServerImpl::Ping(RpcController* cntl, const Empty* req, Empty* resp,
                      Closure* done) {
  brpc::ClosureGuard done_guard(done);
}

void ServerImpl::IndirectPing(RpcController* cntl, const PingReq* req,
                              AckResp* resp, Closure* done) {
  brpc::ClosureGuard done_guard(done);

  Empty forward_req;
  Empty forward_resp;

  if (Client::Send(req->ip(), req->port(), forward_req, &forward_resp)) {
    resp->set_type(AckResp_Type_ACK);
  } else {
    resp->set_type(AckResp_Type_NACK);
  }
}

void ServerImpl::Suspect(RpcController* cntl, const SuspectMsg* req,
                         Empty* resp, Closure* done) {
  brpc::ClosureGuard done_guard(done);
}

void ServerImpl::Sync(RpcController* cntl, const SyncMsg* req, SyncMsg* resp,
                      Closure* done) {
  brpc::ClosureGuard done_guard(done);
}

void ServerImpl::Dead(RpcController* cntl, const DeadMsg* req, Empty* resp,
                      Closure* done) {
  brpc::ClosureGuard done_guard(done);
}

}  // namespace rpc
}  // namespace gossip
}  // namespace common
