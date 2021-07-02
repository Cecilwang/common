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

namespace {

void AppendLog(std::string* dst, const std::string& src) {
  dst->empty() ? (*dst = src) : (*dst += " " + src);
}

}  // namespace

namespace common {
namespace gossip {
namespace rpc {

#define DefineSend(FUNC, REQ, RESP)                               \
  template <>                                                     \
  void Client::Send(ServerAPI_Stub* stub, brpc::Controller* cntl, \
                    const REQ* req, RESP* resp) {                 \
    stub->FUNC(cntl, req, resp, nullptr);                         \
  }

DefineSend(Ping, NodeMsg, NodeMsg);
DefineSend(Forward, ForwardMsg, NodeMsg);
DefineSend(Sync, SyncMsg, SyncMsg);

#undef DefineSend

template <class REQ, class RESP>
bool Client::Send(const net::Address& addr, const REQ& req, RESP* resp,
                  uint64_t timeout_ms, int32_t n_retry) {
  brpc::ChannelOptions options;
  options.timeout_ms = timeout_ms;
  options.max_retry = n_retry;

  brpc::Channel channel;
  if (channel.Init(addr.ip()->ip().c_str(), addr.port(), &options) != 0) {
    LOG(ERROR) << "GossipClient failed to initialize channel";
    return false;
  }

  brpc::Controller cntl;
  // cntl.set_log_id(log_id ++);

  ServerAPI_Stub stub(&channel);

  VLOG(5) << "Sending " << req.ShortDebugString() << " to " << addr;
  Send(&stub, &cntl, &req, resp);

  if (cntl.Failed()) {
    LOG(ERROR) << "Failed to send to " << cntl.remote_side() << ": "
               << cntl.ErrorText();
    return false;
  }
  return true;
}

template bool Client::send(const net::Address& addr, const NodeMsg& req,
                           NodeMsg* resp, uint64_t timeout_ms, int32_t n_retry);

template bool Client::send(const net::Address& addr, const ForwardMsg& req,
                           NodeMsg* resp, uint64_t timeout_ms, int32_t n_retry);

template bool Client::send(const net::Address& addr, const SyncMsg& req,
                           SyncMsg* resp, uint64_t timeout_ms, int32_t n_retry);

//------------------------------------------------------------------------------

ServerImpl::ServerImpl(Cluster* cluster) : cluster_(cluster) {}

void ServerImpl::Ping(RpcController* cntl, const NodeMsg* req, NodeMsg* resp,
                      Closure* done) {
  brpc::ClosureGuard done_guard(done);
  VLOG(5) << "Receiving ping " << req->ShortDebugString();
  cluster_->Recv(req, resp);
}

void ServerImpl::Forward(RpcController* cntl, const ForwardMsg* req,
                         NodeMsg* resp, Closure* done) {
  brpc::ClosureGuard done_guard(done);
  VLOG(5) << "Receiving forward " << req->ShortDebugString();
  net::Address addr(req->ip(), req->port());
  if (!Client::Send(addr, req->node(), resp)) {  // failed
    resp->set_state(State::UNKNOWN);
  }
  // TODO(sxwang): Of course, we can take some measures based on the current
  // situation, but now we don't.
}

void ServerImpl::Sync(RpcController* cntl, const SyncMsg* req, SyncMsg* resp,
                      Closure* done) {
  brpc::ClosureGuard done_guard(done);
  VLOG(5) << "Receiving sync " << req->ShortDebugString();
  // TODO(sxwang): It's much better to avoid flood

  // TODO(sxwang): Need other acitons to handle clutser_->Stop()

  SyncMsg req_back = *req;
  std::thread t([=] { cluster_->RecvSync(&req_back); });
  t.detach();

  cluster_->GenSyncMsg(resp);
}

}  // namespace rpc

//------------------------------------------------------------------------------

BroadcastQueue::BroadcastQueue(uint32_t n_transmit)
    : n_transmit_(n_transmit), queue_(ElementCmp) {}

void BroadcastQueue::Push(Node::ConstPtrRef node) {
  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto search = existence_.find(node);
    if (search != existence_.end()) {  // drop outdated element
      queue_.erase(queue_.find(search->second));
    }

    auto e = std::make_shared<Element>(node, 0, ++id_);
    queue_.insert(e);
    existence_[node] = e;
    LOG(INFO) << "BroadcastQueue pushed " << node->ToString(true);
  }
  cv_.notify_all();
}

Node::Ptr BroadcastQueue::Pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !queue_.empty(); });

  auto head = queue_.begin();
  auto e = *head;
  queue_.erase(head);  // We have to erase then insert to rebalance the tree.
  if (++e->n_transmit < n_transmit_) {
    queue_.insert(e);
    VLOG(5) << "BroadcastQueue repushed " << *e->node << " " << e->n_transmit
            << "|" << n_transmit_;
  } else {
    existence_.erase(e->node);
    VLOG(5) << "BroadcastQueue popped " << *e->node << " " << e->n_transmit
            << "|" << n_transmit_;
  }

  return e->node;
}

size_t BroadcastQueue::Size() {
  std::unique_lock<std::mutex> lock(mutex_);
  size_t ret = queue_.size();
  lock.unlock();
  cv_.notify_all();
  return ret;
}

std::string BroadcastQueue::ToString() {
  std::unique_lock<std::mutex> lock(mutex_);

  std::ostringstream ss;
  ss << "BroadcastQueue transmits in " << n_transmit_ << " times.";
  for (const auto& e : queue_) {
    ss << "\n\t" << e->node->ToString(true) << " " << e->n_transmit << " times";
  }

  lock.unlock();
  cv_.notify_one();
  return ss.str();
}

BroadcastQueue::operator std::string() { return ToString(); }

std::ostream& operator<<(std::ostream& os, BroadcastQueue& self) {
  return os << self.ToString();
}

BroadcastQueue::Element::Element(Node::ConstPtrRef node, uint32_t n_transmit,
                                 uint32_t id)
    : node(node), n_transmit(n_transmit), id(id) {}

bool BroadcastQueue::ElementCmp(Element::ConstPtrRef a,
                                Element::ConstPtrRef b) {
  return a->n_transmit < b->n_transmit ||
         (a->n_transmit == b->n_transmit && a->id > b->id);
}

//------------------------------------------------------------------------------

Cluster::Cluster(uint16_t port, uint32_t n_transmit, uint64_t probe_inv_ms,
                 uint64_t sync_inv_ms, uint64_t gossip_inv_ms)
    : version_(0),
      queue_(n_transmit),
      addr_("0.0.0.0", port),
      probe_inv_ms_(probe_inv_ms),
      sync_inv_ms_(sync_inv_ms),
      gossip_inv_ms_(gossip_inv_ms) {}

Cluster::~Cluster() { Stop(); }

Cluster& Cluster::Alive(const std::string& name) {
  if (!name_.empty()) {
    LOG(WARNING) << ToString() << " has been alive with " << name_;
    return *this;
  }
  // TODO(sxwang) use MAC address instead of hostname.
  name_ = name.empty() ? net::GetHostname() : name;
  rpc::NodeMsg msg;
  msg.set_name(name_);
  msg.set_version(++version_);
  msg.set_ip(net::GetDelegateIP(net::GetPublicIPs(), addr_.ip())->ip());
  msg.set_port(addr_.port());
  msg.set_state(rpc::State::ALIVE);
  msg.set_metadata("");
  Recv(&msg, nullptr);
  return *this;
}

Cluster& Cluster::Start() {
  StartServer() && StartRoutine();
  return *this;
}

bool Cluster::StartServer() {
  if (server_.IsRunning()) {
    LOG(WARNING) << *this << " has listened on " << addr_;
    return false;
  }

  if (server_.AddService(new rpc::ServerImpl(this),
                         brpc::SERVER_OWNS_SERVICE) != 0) {
    LOG(ERROR) << *this << " failed to add service.";
    return false;
  }

  brpc::ServerOptions options;
  options.has_builtin_services = false;

  if (server_.Start(addr_.ToString().c_str(), &options) != 0) {
    LOG(ERROR) << *this << " failed to listen on " << addr_;
    server_.ClearServices();
    return false;
  }

  LOG(INFO) << ToString() << " is listening on " << addr_;
  return true;
}

bool Cluster::StartRoutine() {
  if (!probe_t_) {
    probe_t_ = util::CreateLoopThread([this] { Probe(); }, probe_inv_ms_, true);
    probe_t_->Run();
  }
  if (!sync_t_) {
    sync_t_ = util::CreateLoopThread(
        [this] { Sync(); },
        [this] {
          size_t n = std::max(static_cast<size_t>(32), nodes_v_.size());
          return sync_inv_ms_ * (ceil(log(n) - log(32)) + 1.0);
        },
        true);
    sync_t_->Run();
  }
  if (!gossip_t_) {
    gossip_t_ =
        util::CreateLoopThread([this] { Gossip(); }, gossip_inv_ms_,
                               [this] { return queue_.Size() > 0; }, true);
    gossip_t_->Run();
  }
  return true;
}

Cluster& Cluster::Stop() {
  StopRoutine();
  StopServer();
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

void Cluster::Join(const std::string& ip, uint16_t port) {
  rpc::SyncMsg req = GenSyncMsg();
  rpc::SyncMsg resp;
  rpc::Client::Send(net::Address(ip, port), req, &resp, 10 * 1000);
  RecvSync(&resp);
  LOG(INFO) << *this << " joined " << ip << ":" << port;
}

void Cluster::Probe() {
  // We don't need to guarantee that nodes_v_ is the same in each iteration.
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
          node->state() == rpc::State::DEAD) {
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
  Probe(node);
}

void Cluster::Probe(Node::ConstPtrRef node) {
  uint64_t timeout_ms = health_ * probe_inv_ms_;
  VLOG(5) << *this << " is probing " << *node << " in " << timeout_ms << "ms";

  rpc::NodeMsg resp;

  // Directly ping
  // 1. ALIVE->ack
  // 2. failed->Indirectly ping
  // otherwise invalid
  if (rpc::Client::Send(node->addr(), GenNodeMsg(node), &resp, timeout_ms)) {
    if (resp.state() != rpc::State::ALIVE) {
      LOG(WARNING) << *this << " received invalid ack for ping to " << *node
                   << ", resp: " << resp.ShortDebugString();
      // Something went wrong, exit without taking any action
      return;
    }
    health_ += -1;
    VLOG(5) << *this << " succeed to probe " << *node;
    Recv(&resp, nullptr);
    return;
  }
  LOG(WARNING) << *this << " failed to probe " << *node << " directly.";

  // Use the filter to randomly select up to 3 elements where 3 is arbitrary
  auto nodes = GetRandomNodes(3, [this, &node](Node::ConstPtrRef x) {
    return x->state() != rpc::State::ALIVE ||  //
           x->name() == name_ ||               //
           x->name() == node->name();
  });

  // Indirectly ping
  // 1. ALIVE->ack
  // 2. UNKNOWN->nack
  // 3. failed
  // otherwise invalid
  for (const auto& x : nodes) {
    if (rpc::Client::Send(x->addr(), GenForwardMsg(node), &resp, timeout_ms)) {
      if (resp.state() == rpc::State::ALIVE) {  // ack
        VLOG(5) << *this << " succeed to probe " << *node << " with " << *x;
        health_ += -1;
        Recv(&resp, nullptr);
        return;
      } else if (resp.state() != rpc::State::UNKNOWN) {
        LOG(WARNING) << *this
                     << " received invalid ack for indirect ping. resp: "
                     << resp.ShortDebugString();
        // Something went wrong, but don't take any action
      }
    } else {  // failed to connect;
      LOG(WARNING) << *this << " failed to probe " << *node << " with " << *x;
      health_ += 1;
    }
  }

  // All nack or failed, mark it as suspect
  LOG(WARNING) << *this << " failed to probe " << *node;
  auto msg = GenNodeMsg(node, rpc::State::SUSPECT);
  Recv(&msg, nullptr);
}

void Cluster::Sync() {
  // Use the filter to randomly select up to 1 elements
  auto nodes = GetRandomNodes(1, [this](Node::ConstPtrRef x) {
    return x->state() != rpc::State::ALIVE || x->name() == name_;
  });

  if (nodes.size() == 0) {
    return;
  }

  Node::Ptr node = *(nodes.begin());
  VLOG(5) << *this << " is syncing with " << *node;

  rpc::SyncMsg req = GenSyncMsg();
  rpc::SyncMsg resp;
  rpc::Client::Send(node->addr(), req, &resp, 10 * 1000);
  RecvSync(&resp);
}

void Cluster::Gossip() {
  // Use the filter to randomly select up to 3 elements where 3 is arbitrary
  auto nodes = GetRandomNodes(3, [this](Node::ConstPtrRef x) {
    return x->name() == name_ ||  //
           (x->state() == rpc::State::DEAD && x->elapsed_ms() >= 1 * 60 * 1000);
  });

  size_t n = std::min(queue_.Size(), nodes.size() * 3);

  rpc::NodeMsg resp;
  for (size_t i = 0; i < n; ++i) {
    Node::Ptr dst = nodes[i % nodes.size()];
    Node::Ptr x = queue_.Pop();
    VLOG(5) << *this << " gossips " << *x << " to " << *dst;
    rpc::Client::Send(dst->addr(), GenNodeMsg(x), &resp);
  }
}

void Cluster::Recv(const rpc::NodeMsg* req, rpc::NodeMsg* resp) {
  rpc::NodeMsg unused;
  resp = resp == nullptr ? &unused : resp;
  switch (req->state()) {
    case rpc::State::ALIVE:
      RecvAlive(req, resp);
      break;
    case rpc::State::SUSPECT:
      RecvSuspect(req, resp);
      break;
    case rpc::State::DEAD:
      RecvDead(req, resp);
      break;
    default:
      LOG(WARNING) << *this << " discarded an UNKNOWN NodeMsg.";
      break;
  }
}

void Cluster::RecvAlive(const rpc::NodeMsg* alive, rpc::NodeMsg* resp) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);

  Node::Ptr node = nullptr;
  auto search = nodes_m_.find(alive->name());
  if (search == nodes_m_.end()) {  // New Node
    node = std::make_shared<Node>(alive);
    nodes_m_[alive->name()] = node;
    nodes_v_.push_back(node);
    size_t n = nodes_v_.size();
    // Randomly select an element from [0, and-1] and exchange it with the
    // latest element. Of course, nothing may happen, but we can't use [0, n-2]
    // instead of [0, n-1] because n may be equal to 1.
    std::swap(nodes_v_[util::Uniform(0, n - 1)], nodes_v_[n - 1]);
    LOG(INFO) << *this << " received new ALIVE: " << *node;
  } else {
    node = search->second;
  }

  if (alive->name() == name_) {      // about me
    if (search == nodes_m_.end()) {  // internal bootstrap
      LOG(INFO) << *this << " started itself: " << *node << ".";
      Broadcast(node);
    } else if (*node < *alive ||
               (node->version() == alive->version() && *node != *alive)) {
      // refute external message
      // 1. I have reset in the past
      // 2. The external message conflicts with me.
      Refute(node, alive->version(), resp);
      LOG(INFO) << *this << " refutes the rumor about me.";
    } else {  // send latest message about me back
      VLOG(5) << *this << " syncs my latest information with peer.";
      GenNodeMsg(node, resp);
    }
  } else {  // about others
    // We only handle the new/reset message and discard others.
    if (*node < *alive || node->Reset(alive)) {
      *node = *alive;
      LOG(INFO) << *this << " received new ALIVE message " << *node;
      Broadcast(node);
    }
  }
}

void Cluster::RecvSuspect(const rpc::NodeMsg* suspect, rpc::NodeMsg* resp) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);

  auto search = nodes_m_.find(suspect->name());
  if (search == nodes_m_.end()) {
    return;
  }

  auto node = search->second;

  if (*node > *suspect) {  // Outdated message
    return;
  }

  if (node->state() != rpc::State::ALIVE) {
    if (node->state() == rpc::State::SUSPECT &&
        node->suspect_timer()->AddSuspector(suspect->from())) {
      // received new suspector
      LOG(INFO) << *this << ":" << *node << " received new suspector "
                << suspect->from();
      Broadcast(node);
    }
    // nothing need to do
    return;
  }

  if (node->name() == name_) {  // about me
    LOG(INFO) << *this << " refute the rumor about suspection.";
    Refute(node, suspect->version(), resp);
    return;
  }

  // ALIVE->DEAD
  *node = *suspect;

  // Allow the following numbers are arbitrary
  // 2+2 -> (suspect + me) + 2 other nodes;
  // 1+2 -> me + 2 other nodes
  size_t n = nodes_m_.size() <= 2 + 2 ? 0 : 1 + 2;
  uint64_t min_ms = 4 * std::max(1.0, log10(nodes_m_.size())) * probe_inv_ms_;
  uint64_t max_ms = 6 * min_ms;
  // Setup suspect timer
  node->set_suspect_timer(std::make_shared<SuspectTimer>(
      util::CreateTimer(
          [this, node] {
            // We must detach another thread to run the callback in order to
            // avoiding deadlock. Otherwise, RecvDead will try to delete timer
            // in the callback of timer. Justing image that this DAED message is
            // from void.
            std::thread([this, node] {
              // timeout reached, SUSPECT->DEAD
              LOG(INFO) << *this << ":" << *node << "'s suspect timer is up.";
              auto msg = GenNodeMsg(node, rpc::State::DEAD);
              Recv(&msg, nullptr);
            }).detach();
          },
          n == 0 ? min_ms : max_ms),
      n, min_ms, max_ms, name_));

  Broadcast(node);
}

void Cluster::RecvDead(const rpc::NodeMsg* dead, rpc::NodeMsg* resp) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);

  auto search = nodes_m_.find(dead->name());
  if (search == nodes_m_.end()) {
    return;
  }

  auto node = search->second;

  if (*node > *dead ||                      // Outdated message
      node->state() == rpc::State::DEAD) {  // duplicate
    return;
  }

  if (dead->name() == name_ && dead->from() != name_) {  // rumor about me
    Refute(node, dead->version(), resp);
    return;
  }

  // ALIVE/SUSPECT->DEAD include self-shutdown.
  *node = *dead;
  LOG(INFO) << *this << " received new DEAD: " << *node;

  Broadcast(node);
}

void Cluster::RecvSync(const rpc::SyncMsg* req) {
  for (const auto& x : req->nodes()) {
    Recv(&x, nullptr);
  }
}

void Cluster::Refute(Node::Ptr node, uint32_t version, rpc::NodeMsg* resp) {
  version_ += version - version_ + 1;
  health_ += 1;
  node->set_version(version_);
  // Send the refute to the sender directly
  GenNodeMsg(node, resp);
  Broadcast(node);
}

void Cluster::Broadcast(Node::ConstPtrRef node) { queue_.Push(node); }

void Cluster::GenNodeMsg(Node::ConstPtrRef src, rpc::NodeMsg* dst) const {
  src->ToNodeMsg(dst);
  dst->set_from(name_);
}

rpc::NodeMsg Cluster::GenNodeMsg(Node::ConstPtrRef node) const {
  rpc::NodeMsg msg;
  GenNodeMsg(node, &msg);
  return msg;
}

rpc::NodeMsg Cluster::GenNodeMsg(Node::ConstPtrRef node,
                                 rpc::State state) const {
  rpc::NodeMsg msg = GenNodeMsg(node);
  msg.set_state(state);
  return msg;
}

rpc::ForwardMsg Cluster::GenForwardMsg(Node::ConstPtrRef node) const {
  rpc::ForwardMsg msg;
  msg.set_ip(node->ip());
  msg.set_port(node->port());
  GenNodeMsg(node, msg.mutable_node());
  return msg;
}

rpc::ForwardMsg Cluster::GenForwardMsg(Node::ConstPtrRef node,
                                       rpc::State state) const {
  rpc::ForwardMsg msg = GenForwardMsg(node);
  msg.mutable_node()->set_state(state);
  return msg;
}

void Cluster::GenSyncMsg(rpc::SyncMsg* msg) {
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  for (const auto& x : nodes_v_) {
    auto node_msg = msg->add_nodes();
    GenNodeMsg(x, node_msg);
    if (node_msg->state() == rpc::State::DEAD) {
      node_msg->set_state(rpc::State::SUSPECT);
    }
  }
  VLOG(5) << *this << " generated SyncMsg.";
}

rpc::SyncMsg Cluster::GenSyncMsg() {
  rpc::SyncMsg msg;
  GenSyncMsg(&msg);
  return msg;
}

void Cluster::ShuffleNodes() {
  std::string log;
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  size_t n = nodes_v_.size();
  for (size_t i = 0; i < n;) {
    if (nodes_v_[i]->state() == rpc::State::DEAD &&
        nodes_v_[i]->elapsed_ms() >= 1 * 60 * 1000) {
      AppendLog(&log, nodes_v_[i]->ToString());
      nodes_m_.erase(nodes_v_[i]->name());
      std::swap(nodes_v_[i], nodes_v_[n - 1]);
      nodes_v_.pop_back();
      --n;
    } else {
      ++i;
    }
  }
  LOG_IF(INFO, !log.empty()) << *this << " kicked off " << log;
  std::random_shuffle(nodes_v_.begin(), nodes_v_.end());
}

template <class F>
std::vector<Node::Ptr> Cluster::GetRandomNodes(size_t maxn, F&& f) {
  std::unordered_set<Node::Ptr> nodes;
  std::string log;
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  size_t n = nodes_v_.size();
  for (size_t i = 0, j = 0; i < n && j < maxn; ++i) {
    auto node = nodes_v_[util::Uniform(0, n - 1)];
    if (!f(node) && nodes.find(node) == nodes.end()) {
      ++j;
      nodes.insert(node);
      AppendLog(&log, node->ToString());
    }
  }
  VLOG(5) << *this << " randomly selected " << log;
  return std::vector<Node::Ptr>(nodes.begin(), nodes.end());
}

std::vector<Node::Ptr> Cluster::Nodes() {
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  std::vector<Node::Ptr> ret;
  for (const auto& x : nodes_v_) {
    ret.push_back(std::make_shared<Node>(x->name(), x->version(), x->ip(),
                                         x->port(), x->state(), x->metadata()));
  }
  return ret;
}

std::string Cluster::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Cluster(version: " << version_
       << " state: " << (server_.IsRunning() ? "up" : "down")
       << ", address: " << addr_.ToString() << ")";
    auto nodes = const_cast<Cluster*>(this)->Nodes();
    for (const auto& x : nodes) {
      ss << std::endl << x->ToString(true);
    }
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
