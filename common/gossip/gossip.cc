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

#include "common/gossip/gossip.h"

#include "glog/logging.h"

namespace common {
namespace gossip {

void GossipServerImpl::Ping(::google::protobuf::RpcController* cntl,
                            const PingReq* req, PingResp* resp,
                            ::google::protobuf::Closure* done) {
  resp->set_type(PingResp_Type_ACK);
  done->Run();
}

void GossipServerImpl::IndirectPing(::google::protobuf::RpcController* cntl,
                                    const IndirectPingReq* req, PingResp* resp,
                                    ::google::protobuf::Closure* done) {
  done->Run();
}

void GossipServerImpl::Sync(::google::protobuf::RpcController* cntl,
                            const SyncMsg* req, SyncMsg* resp,
                            ::google::protobuf::Closure* done) {
  done->Run();
}

void GossipServerImpl::Suspect(::google::protobuf::RpcController* cntl,
                               const SuspectMsg* req,
                               google::protobuf::Empty* resp,
                               ::google::protobuf::Closure* done) {
  done->Run();
}

void GossipServerImpl::Dead(::google::protobuf::RpcController* cntl,
                            const DeadMsg* req, google::protobuf::Empty* resp,
                            ::google::protobuf::Closure* done) {
  done->Run();
}

//------------------------------------------------------------------------------

GossipClientImpl::GossipClientImpl()
    : client_(sofa::pbrpc::RpcClientOptions()) {}

std::unique_ptr<PingResp> GossipClientImpl::Ping(const net::Address& address,
                                                 uint64_t timeout) {
  sofa::pbrpc::RpcChannel channel(&client_, address.ToString());
  GossipServerAPI_Stub stub(&channel);

  sofa::pbrpc::RpcController cntl;
  cntl.SetTimeout(timeout);

  PingReq req;
  std::unique_ptr<PingResp> resp(new PingResp());

  stub.Ping(&cntl, &req, resp.get(), nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "Failed to call Ping to " << cntl.RemoteAddress().c_str()
               << ": errcode[" << cntl.ErrorCode() << "], errMessage["
               << cntl.ErrorText() << "]";
    return nullptr;
  }

  return resp;
}

//------------------------------------------------------------------------------

Cluster::Cluster(uint16_t port, int32_t n_worker)
    : address_("0.0.0.0", port), n_worker_(n_worker) {}

Cluster::~Cluster() { Stop(); }

void Cluster::Start() {
  if (server_) {
    LOG(WARNING) << ToString() << " has listened on " << address_;
    return;
  }

  sofa::pbrpc::RpcServerOptions option;
  option.work_thread_num = n_worker_;
  server_.reset(new sofa::pbrpc::RpcServer(option));

  // sofa::pbrpc::RpcServer will take the ownership of the GossipServerImpl, and
  // the GossipServerImpl will be deleted when sofa::pbrpc::RpcServer calls
  // Stop().
  if (!server_->RegisterService(new GossipServerImpl())) {
    LOG(ERROR) << "Failed to register rpc services to cluster";
    return;
  }

  if (!server_->Start(address_.ToString())) {
    LOG(ERROR) << ToString() << " failed to listen on " << address_.ToString();
    return;
  }
  LOG(INFO) << ToString() << " is listening on " << address_.ToString();
}

void Cluster::Stop() {
  if (server_) {
    server_->Stop();
    server_.reset(nullptr);
  }
}

std::string Cluster::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Cluster(state: " << (server_ ? "up" : "down")
       << ", address: " << address_.ToString() << ", self:" << self_ << ")";
  } else {
    ss << "Cluster(" << address_.ToString() << ")";
  }
  return ss.str();
}

Cluster::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Cluster& self) {
  return os << self.ToString();
}

}  // namespace gossip
}  // namespace common
