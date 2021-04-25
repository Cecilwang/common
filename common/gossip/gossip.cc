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
namespace rpc {

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

}  // namespace rpc

//------------------------------------------------------------------------------

Node::Node(uint32_t version, const std::string& name, const std::string& ip,
           uint16_t port, rpc::State state, const std::string& metadata)
    : version_(version),
      name_(name),
      ip_(ip),
      port_(port),
      state_(state),
      metadata_(metadata) {}

Node::Node(const rpc::AliveMsg* alive)
    : version_(alive->version()),
      name_(alive->name()),
      ip_(alive->ip()),
      port_(alive->port()),
      state_(rpc::State::ALIVE),
      metadata_(alive->metadata()) {}

Node& Node::operator=(const rpc::AliveMsg& alive) {
  if (name_ != alive.name()) {
    return *this;
  }
  version_ = alive.version();
  ip_ = alive.ip();
  port_ = alive.port();
  state_ = rpc::State::ALIVE;
  metadata_ = alive.metadata();
  return *this;
}

bool Node::Conflict(const rpc::AliveMsg* alive) const {
  return (name_ == alive->name()) &&
         (ip_ != alive->ip() || port_ != alive->port() ||
          metadata_ != alive->metadata());
}

bool Node::Reset(const rpc::AliveMsg* alive) const {
  // Here, we can only recognize the reset by the content. However, of course,
  // sometimes the node will reset without any changes. In these cases, reset
  // node will refute others outdated message in the future. Even if the delay
  // is uncertain, synchronization will eventually occur.
  // is uncertain, the synchronize will happen eventually.
  return (state_ == rpc::State::DEAD || state_ == rpc::State::LEFT) &&
         Conflict(alive);
}

uint32_t Node::version() const { return version_; }

const std::string& Node::name() const { return name_; }

const std::string& Node::ip() const { return ip_; }

uint16_t Node::port() const { return port_; }

rpc::State Node::state() const { return state_; }

const std::string& Node::metadata() const { return metadata_; }

std::string Node::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Node(" << version_ << ", " << name_ << ", " << ip_ << ", " << port_
       << ")";
  } else {
    ss << "Node(" << version_ << ", " << name_ << ")";
  }
  return ss.str();
}

Node::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Node& self) {
  return os << self.ToString();
}

bool operator>(const Node& node, const rpc::AliveMsg& alive) {
  return node.name() == alive.name() && node.version() > alive.version();
}

bool operator<=(const Node& node, const rpc::AliveMsg& alive) {
  return node.name() == alive.name() && node.version() <= alive.version();
}

bool operator==(const Node& node, const rpc::AliveMsg& alive) {
  return node.version() == alive.version() && node.name() == alive.name() &&
         node.ip() == alive.ip() && node.port() == alive.port() &&
         node.state() == rpc::State::ALIVE &&
         node.metadata() == alive.metadata();
}

//------------------------------------------------------------------------------

Cluster::Cluster(uint16_t port, int32_t n_worker)
    : version_(0), address_("0.0.0.0", port), n_worker_(n_worker) {}

Cluster::~Cluster() { Stop(); }

Cluster& Cluster::Start() {
  if (server_) {
    LOG(WARNING) << ToString() << " has listened on " << address_;
    return *this;
  }

  sofa::pbrpc::RpcServerOptions option;
  option.work_thread_num = n_worker_;
  server_.reset(new sofa::pbrpc::RpcServer(option));

  // sofa::pbrpc::RpcServer will take the ownership of the GossipServerImpl, and
  // the GossipServerImpl will be deleted when sofa::pbrpc::RpcServer calls
  // Stop().
  if (!server_->RegisterService(new rpc::GossipServerImpl())) {
    LOG(ERROR) << "Failed to register rpc services to cluster";
    return *this;
  }

  if (!server_->Start(address_.ToString())) {
    LOG(ERROR) << ToString() << " failed to listen on " << address_.ToString();
    return *this;
  }
  LOG(INFO) << ToString() << " is listening on " << address_.ToString();
  return *this;
}

Cluster& Cluster::Stop() {
  if (server_) {
    server_->Stop();
    server_.reset(nullptr);
  }
  return *this;
}

Cluster& Cluster::Alive() {
  rpc::AliveMsg alive;
  alive.set_version(++version_);
  alive.set_name(net::GetHostname());
  alive.set_ip(net::GetDelegateIP(net::GetPublicIPs(), address_.ip())->ip());
  alive.set_port(address_.port());
  alive.set_metadata("");
  RecvAlive(&alive);
  return *this;
}

void Cluster::RecvAlive(rpc::AliveMsg* alive) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);

  std::shared_ptr<Node> node = nullptr;
  auto search = nodes_.find(alive->name());
  if (search == nodes_.end()) {  // New Node
    node = std::make_shared<Node>(alive);
    nodes_[alive->name()] = node;
  } else {
    node = search->second;
  }

  if (strcmp(alive->name().c_str(), net::GetHostname()) == 0) {  // myself
    if (search == nodes_.end()) {  // internal boostrap
      LOG(INFO) << *this << " started itself: " << *node << ".";
      // Broadcast(alive);
    } else {                                    // external message
      if (*node > *alive || *node == *alive) {  // outdated message
        return;
      }
      // Refute will also make resetting work.
      // Refute(alive);
    }
  } else {  // otherself
    // We only handle the reset and the new message and discard the conflict
    // message.
    if (node->Reset(alive) || *node <= *alive) {
      auto log = node->ToString();
      *node = *alive;
      LOG(INFO) << log << " received(" << alive->DebugString() << ") -> "
                << *node << ".";
      // ClearTimer(node);
      // Broadcast(alive);
    }
  }
}

std::string Cluster::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Cluster(version: " << version_
       << " state: " << (server_ ? "up" : "down")
       << ", address: " << address_.ToString() << ")";
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
