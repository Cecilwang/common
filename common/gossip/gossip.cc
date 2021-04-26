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

#include <utility>

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
                               const SuspectReq* req,
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

template <class REQ, class RESP>
void CallFunc(GossipServerAPI_Stub*, sofa::pbrpc::RpcController*, const REQ*,
              RESP*);

template <>
void CallFunc(GossipServerAPI_Stub* stub, sofa::pbrpc::RpcController* cntl,
              const PingReq* req, PingResp* resp) {
  stub->Ping(cntl, req, resp, nullptr);
}

template <>
void CallFunc(GossipServerAPI_Stub* stub, sofa::pbrpc::RpcController* cntl,
              const SuspectReq* req, ::google::protobuf::Empty* resp) {
  stub->Suspect(cntl, req, resp, nullptr);
}

GossipClientImpl::GossipClientImpl()
    : client_(sofa::pbrpc::RpcClientOptions()) {}

template <class REQ, class RESP>
bool GossipClientImpl::Send(const net::Address& addr, const REQ& req,
                            RESP* resp, uint64_t timeout) {
  sofa::pbrpc::RpcChannel channel(&client_, addr.ToString());
  GossipServerAPI_Stub stub(&channel);

  sofa::pbrpc::RpcController cntl;
  cntl.SetTimeout(timeout);

  CallFunc(&stub, &cntl, &req, resp);

  if (cntl.Failed()) {
    LOG(ERROR) << "Failed to call Ping to " << cntl.RemoteAddress().c_str()
               << ": errcode[" << cntl.ErrorCode() << "], errMessage["
               << cntl.ErrorText() << "]";
    return false;
  }
  return true;
}

}  // namespace rpc

//------------------------------------------------------------------------------

int Health::score() {
  std::lock_guard<std::mutex> lock(mutex_);
  return score_;
}

Health& Health::operator+=(int val) {
  std::lock_guard<std::mutex> lock(mutex_);
  score_ = std::min(std::max(1, score_ + val), upper_);
  return *this;
}

int Health::operator*(int val) {
  std::lock_guard<std::mutex> lock(mutex_);
  return score_ * val;
}

//------------------------------------------------------------------------------

Node::Node(uint32_t version, const std::string& name, const std::string& ip,
           uint16_t port, rpc::State state, const std::string& metadata)
    : version_(version),
      name_(name),
      addr_(ip, port),
      state_(state),
      metadata_(metadata) {}

Node::Node(const rpc::AliveMsg* alive)
    : version_(alive->version()),
      name_(alive->name()),
      addr_(alive->ip(), alive->port()),
      state_(rpc::State::ALIVE),
      metadata_(alive->metadata()) {}

Node& Node::operator=(const rpc::AliveMsg& alive) {
  if (name_ != alive.name()) {
    return *this;
  }
  version_ = alive.version();
  addr_.set_ip(alive.ip());
  addr_.set_port(alive.port());
  state_ = rpc::State::ALIVE;
  metadata_ = alive.metadata();
  return *this;
}

bool Node::Conflict(const rpc::AliveMsg* alive) const {
  return (name_ == alive->name()) &&
         (ip() != alive->ip() || port() != alive->port() ||
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

const net::Address& Node::ToAddress() const { return addr_; }

uint32_t Node::version() const { return version_; }

void Node::set_version(uint32_t version) { version_ = version; }

const std::string& Node::name() const { return name_; }

const std::string& Node::ip() const { return addr_.ip()->ip(); }

uint16_t Node::port() const { return addr_.port(); }

rpc::State Node::state() const { return state_; }

const std::string& Node::metadata() const { return metadata_; }

std::string Node::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Node(" << version_ << ", " << name_ << ", " << addr_ << ")";
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

BroadcastQueue::BroadcastQueue(uint32_t n_transmit)
    : n_transmit_(n_transmit), queue_(ElementCmp) {}

void BroadcastQueue::Push(Node::ConstPtr node) {
  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto search = existence_.find(node);
    if (search != existence_.end()) {  // drop outdated element
      queue_.erase(queue_.find(search->second));
    }

    auto e = std::make_shared<Element>(node, 0, ++id_);
    queue_.insert(e);
    existence_[node] = e;
  }
  cv_.notify_one();
}

Node::ConstPtr BroadcastQueue::Pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !queue_.empty(); });

  auto head = queue_.begin();
  auto e = *head;
  queue_.erase(head);  // We have to erase then insert to rebalance the tree.
  if (++e->n_transmit < n_transmit_) {
    queue_.insert(e);
  } else {
    existence_.erase(e->node);
  }
  return e->node;
}

size_t BroadcastQueue::Size() {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

BroadcastQueue::Element::Element(Node::ConstPtr node, uint32_t n_transmit,
                                 uint32_t id)
    : node(node), n_transmit(n_transmit), id(id) {}

bool BroadcastQueue::ElementCmp(Element::ConstPtr a, Element::ConstPtr b) {
  return a->n_transmit < b->n_transmit ||
         (a->n_transmit == b->n_transmit && a->id > b->id);
}

//------------------------------------------------------------------------------

Cluster::Cluster(uint16_t port, int32_t n_worker, uint32_t n_transmit,
                 uint64_t probe_inv_ms, uint64_t sync_inv_ms,
                 uint64_t gossip_inv_ms)
    : version_(0),
      probe_inv_ms_(probe_inv_ms),
      sync_inv_ms_(sync_inv_ms),
      gossip_inv_ms_(gossip_inv_ms),
      addr_("0.0.0.0", port),
      n_worker_(n_worker),
      queue_(n_transmit) {}

Cluster::~Cluster() { Stop(); }

Cluster& Cluster::Alive(const std::string& name) {
  if (!name_.empty()) {
    LOG(WARNING) << ToString() << " has been alive with " << name_;
    return *this;
  }
  name_ = name.empty() ? net::GetHostname() : name;
  rpc::AliveMsg alive;
  alive.set_version(++version_);
  alive.set_name(name_);
  alive.set_ip(net::GetDelegateIP(net::GetPublicIPs(), addr_.ip())->ip());
  alive.set_port(addr_.port());
  alive.set_metadata("");
  RecvAlive(&alive);
  return *this;
}

Cluster& Cluster::Start() {
  StartServer() && StartRoutine();
  return *this;
}

bool Cluster::StartServer() {
  if (server_) {
    LOG(WARNING) << ToString() << " has listened on " << addr_;
    return false;
  }

  sofa::pbrpc::RpcServerOptions option;
  option.work_thread_num = n_worker_;
  server_.reset(new sofa::pbrpc::RpcServer(option));

  // sofa::pbrpc::RpcServer will take the ownership of the GossipServerImpl, and
  // the GossipServerImpl will be deleted when sofa::pbrpc::RpcServer calls
  // Stop().
  if (!server_->RegisterService(new rpc::GossipServerImpl())) {
    LOG(ERROR) << "Failed to register rpc services to cluster";
    server_.reset(nullptr);
    return false;
  }

  if (!server_->Start(addr_.ToString())) {
    LOG(ERROR) << ToString() << " failed to listen on " << addr_.ToString();
    server_.reset(nullptr);
    return false;
  }

  LOG(INFO) << ToString() << " is listening on " << addr_.ToString();
  return true;
}

bool Cluster::StartRoutine() {
  if (!probe_t_) {
    probe_t_ = util::CreateLoopThread([this] { Probe(); }, probe_inv_ms_, true);
  }
  if (!sync_t_) {
    sync_t_ = util::CreateLoopThread([this] { Sync(); }, sync_inv_ms_, true);
  }
  if (!gossip_t_) {
    gossip_t_ =
        util::CreateLoopThread([this] { Gossip(); }, gossip_inv_ms_, true);
  }
  return true;
}

Cluster& Cluster::Stop() {
  StopServer();
  StopRoutine();
  return *this;
}

void Cluster::StopServer() {
  if (server_) {
    server_->Stop();
    server_.reset(nullptr);
    LOG(INFO) << *this << " stoped server.";
  }
}

void Cluster::StopRoutine() {
  if (probe_t_) {
    probe_t_->Stop();
  }
  if (sync_t_) {
    sync_t_->Stop();
  }
  if (gossip_t_) {
    gossip_t_->Stop();
  }
}

void Cluster::Probe() {
  // We don’t need to guarantee that nodes_v_ is the same in each iteration.
  Node::Ptr node = nullptr;
  for (size_t n_checked = 0; !node; ++n_checked) {
    nodes_mutex_.lock();

    size_t n = nodes_v_.size();
    if (n_checked >= n) {  // Have checked all nodes in this round.
      nodes_mutex_.unlock();
      return;
    }

    if (probe_i_ < n) {  // Try to fetch a node
      node = nodes_v_[probe_i_];
      if (node->name() == name_ ||  // self
          node->state() == rpc::State::DEAD ||
          node->state() == rpc::State::LEFT) {
        node = nullptr;  // Skip this one
      }
      nodes_mutex_.unlock();
      ++probe_i_;
    } else {  // at the end
      nodes_mutex_.unlock();
      ShuffleNodes();  // Kick off dead nodes then shuffle
      probe_i_ = 0;
    }
  }

  // TODO(sxwang)
  SendProbe(node);
}

void Cluster::Sync() {}

void Cluster::Gossip() {}

void Cluster::SendProbe(Node::ConstPtr node) {
  uint64_t timeout = health_ * probe_inv_ms_;
  if ((node->state() == rpc::State::ALIVE && SendPing(node, timeout)) ||
      (node->state() != rpc::State::ALIVE && SendSuspect(node, timeout))) {
    health_ += -1;
    return;
  }
  nodes_mutex_.lock();
  auto nodes = GetRandomNodes(3, [this, &node](Node::ConstPtr other) {
    return other->name() == name_ || other->name() == node->name() ||
           other->state() != rpc::State::ALIVE;
  });
  nodes_mutex_.unlock();
  for (const auto& x : nodes) {
    int ret = SendIndirectPing(x, timeout);
    if (ret == 0) {  // ack
      health_ += -1;
      break;
    } else if (ret == -1) {  // failed
      health_ += 1;
    }  // nack
  }
  rpc::SuspectReq suspect;
  suspect.set_version(node->version());
  suspect.set_dst(node->name());
  suspect.add_srcs(name_);
  RecvSuspect(&suspect);
}

bool Cluster::SendPing(Node::ConstPtr node, uint64_t timeout) {
  rpc::PingReq req;
  req.set_dst(node->name());
  req.set_src(name_);
  rpc::PingResp resp;
  return client_.Send(node->ToAddress(), req, &resp, timeout);
}

bool Cluster::SendSuspect(Node::ConstPtr node, uint64_t timeout) {
  rpc::SuspectReq req;
  req.set_version(node->version());
  req.set_dst(node->name());
  req.add_srcs(name_);
  ::google::protobuf::Empty resp;
  return client_.Send(node->ToAddress(), req, &resp, timeout);
}

int Cluster::SendIndirectPing(Node::ConstPtr node, uint64_t timeout) {
  return -1;
}

void Cluster::ShuffleNodes() {
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  size_t n = nodes_v_.size();
  for (size_t i = 0; i < n;) {
    if (nodes_v_[i]->state() == rpc::State::DEAD /* && dead for enough time*/) {
      nodes_m_.erase(nodes_v_[i]->name());
      std::swap(nodes_v_[i], nodes_v_[n - 1]);
      nodes_v_.pop_back();
      --n;
    } else {
      ++i;
    }
  }
  std::random_shuffle(nodes_v_.begin(), nodes_v_.end());
}

template <class F>
std::unordered_set<Node::Ptr> Cluster::GetRandomNodes(size_t k, F&& f) {
  std::unordered_set<Node::Ptr> ret;
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  size_t n = nodes_v_.size();
  for (size_t i = 0, j = 0; j < k && i < n; ++i) {
    auto node = nodes_v_[util::Uniform(0, n - 1)];
    if (!f(node) && ret.find(node) == ret.end()) {
      ++j;
      ret.insert(node);
    }
  }
  return ret;
}

void Cluster::RecvAlive(rpc::AliveMsg* alive) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);

  Node::Ptr node = nullptr;
  auto search = nodes_m_.find(alive->name());
  if (search == nodes_m_.end()) {  // New Node
    node = std::make_shared<Node>(alive);
    nodes_m_[alive->name()] = node;
    nodes_v_.push_back(node);
    size_t n = nodes_v_.size();
    std::swap(nodes_v_[util::Uniform(0, n - 1)], nodes_v_[n - 1]);
  } else {
    node = search->second;
  }

  if (alive->name() == name_) {      // myself
    if (search == nodes_m_.end()) {  // internal boostrap
      LOG(INFO) << *this << " started itself: " << *node << ".";
      Broadcast(node);
    } else {                                    // external message
      if (*node > *alive || *node == *alive) {  // outdated message
        return;
      }
      // Refute will also make resetting work.
      Refute(node, alive->version());
    }
  } else {  // otherself
    // We only handle the reset/new message and discard the conflict message.
    if (node->Reset(alive) || *node <= *alive) {
      auto log = node->ToString();
      *node = *alive;
      LOG(INFO) << log << " received(" << alive->DebugString() << ") -> "
                << *node << ".";
      // TODO(sxwang) ClearTimer(node);
      Broadcast(node);
    }
  }
}

void Cluster::RecvSuspect(rpc::SuspectReq* suspect) {}

void Cluster::Refute(Node::Ptr self, uint32_t version) {
  version_ += version - version_ + 1;
  self->set_version(version_);
  health_ += 1;
  Broadcast(self);
}

void Cluster::Broadcast(Node::ConstPtr node) { queue_.Push(node); }

std::string Cluster::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Cluster(version: " << version_
       << " state: " << (server_ ? "up" : "down")
       << ", address: " << addr_.ToString() << ")";
  } else {
    ss << "Cluster(" << addr_.ToString() << ")";
  }
  return ss.str();
}

Cluster::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Cluster& self) {
  return os << self.ToString();
}

}  // namespace gossip
}  // namespace common
