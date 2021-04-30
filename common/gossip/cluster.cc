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

#include "common/gossip/cluster.h"

#include <algorithm>
#include <utility>

#include "glog/logging.h"

namespace common {
namespace gossip {

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

Cluster::Cluster(uint16_t port, uint32_t n_transmit, uint64_t probe_inv_ms,
                 uint64_t sync_inv_ms, uint64_t gossip_inv_ms)
    : version_(0),
      probe_inv_ms_(probe_inv_ms),
      sync_inv_ms_(sync_inv_ms),
      gossip_inv_ms_(gossip_inv_ms),
      addr_("0.0.0.0", port),
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
  if (server_.IsRunning()) {
    LOG(WARNING) << ToString() << " has listened on " << addr_;
    return false;
  }

  if (server_.AddService(&(rpc::ServerImpl::Get()),
                         brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Failed to add service to cluster";
    return false;
  }

  if (server_.Start(addr_.ToString().c_str(), nullptr) != 0) {
    LOG(ERROR) << ToString() << " failed to listen on " << addr_.ToString();
    server_.ClearServices();
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
  if (server_.IsRunning()) {
    server_.Stop(0);
    server_.Join();
    server_.ClearServices();
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
    int ret = SendIndirectPing(x, node, timeout);
    if (ret == 0) {  // ack
      health_ += -1;
      return;
    } else if (ret == -1) {  // failed
      health_ += 1;
    }  // nack
  }
  // All nack
  rpc::SuspectMsg suspect;
  suspect.set_version(node->version());
  suspect.set_dst(node->name());
  suspect.set_src(name_);
  RecvSuspect(&suspect);
}

bool Cluster::SendPing(Node::ConstPtr node, uint64_t timeout) {
  rpc::Empty req;
  rpc::Empty resp;
  return rpc::Client::Send(node->ip(), node->port(), req, &resp, timeout, 0);
}

int Cluster::SendIndirectPing(Node::ConstPtr broker, Node::ConstPtr target,
                              uint64_t timeout) {
  rpc::PingReq req;
  req.set_ip(target->ip());
  req.set_port(target->port());
  rpc::AckResp resp;
  if (rpc::Client::Send(broker->ip(), broker->port(), req, &resp, timeout, 0)) {
    return resp.type() == rpc::AckResp_Type_ACK ? 0 : 1;
  } else {
    return -1;
  }
}

bool Cluster::SendSuspect(Node::ConstPtr node, uint64_t timeout) {
  rpc::SuspectMsg req;
  req.set_version(node->version());
  req.set_dst(node->name());
  req.set_src(name_);
  rpc::Empty resp;
  return rpc::Client::Send(node->ip(), node->port(), req, &resp, timeout, 0);
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
      Broadcast(node);
    }
  }
}

void Cluster::RecvSuspect(rpc::SuspectMsg* suspect) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);

  auto search = nodes_m_.find(suspect->dst());
  if (search == nodes_m_.end()) {
    return;
  }

  auto node = search->second;

  if (suspect->version() < node->version()) {  // Outdated message
    return;
  }

  if (node->state() != rpc::State::ALIVE) {
    if (node->state() == rpc::State::SUSPECT &&  // received new suspector
        node->suspect_timer()->AddSuspector(suspect->src())) {
      Broadcast(node);
    }
    // nothing need to do
    return;
  }

  if (node->name() == name_) {  // myself
    Refute(node, suspect->version());
    return;
  }

  // ALIVE->DEAD
  *node = *suspect;

  // Allow the following numbers are arbitrary
  // 2+2 -> (suspecteee + myself) + 2 other nodes;
  // 1+2 -> myself + 2 other nodes
  size_t n = nodes_m_.size() <= 2 + 2 ? 0 : 1 + 2;
  uint64_t min_ms = 4 * std::max(1.0, log10(nodes_m_.size())) * probe_inv_ms_;
  uint64_t max_ms = 6 * min_ms;
  // Setup suspect timer
  node->set_suspect_timer(
      std::make_shared<SuspectTimer>(util::CreateTimer(
                                         [this, &node] {
                                           // timeout reached, SUSPECT->DEAD
                                           rpc::DeadMsg dead;
                                           dead.set_version(node->version());
                                           dead.set_dst(node->name());
                                           dead.set_src(name_);
                                           RecvDead(&dead);
                                         },
                                         n == 0 ? min_ms : max_ms),
                                     n, min_ms, max_ms, name_));

  Broadcast(node);
}

void Cluster::RecvDead(rpc::DeadMsg* dead) {}

void Cluster::Refute(Node::Ptr self, uint32_t version) {
  version_ += version - version_ + 1;
  self->set_version(version_);
  health_ += 1;
  Broadcast(self);
}

void Cluster::Broadcast(Node::ConstPtr node) { queue_.Push(node); }

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

std::string Cluster::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Cluster(version: " << version_
       << " state: " << (server_.IsRunning() ? "up" : "down")
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
