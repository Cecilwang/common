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

#include "common/gossip/node.h"

#include <algorithm>
#include <utility>

namespace common {
namespace gossip {

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

SuspectTimer::SuspectTimer(std::unique_ptr<util::Timer> timer, size_t n,
                           uint64_t min_ms, uint64_t max_ms,
                           const std::string& suspector)
    : timer_(std::move(timer)), n_(n), min_ms_(min_ms), max_ms_(max_ms) {
  suspectors_.insert(suspector);
}

bool SuspectTimer::AddSuspector(const std::string& suspector) {
  if (suspectors_.size() >= n_ ||
      suspectors_.find(suspector) != suspectors_.end()) {
    return false;
  }
  suspectors_.insert(suspector);
  size_t m = suspectors_.size();
  timer_->set_timeout_ms(max_ms_ - log(m) / log(n_) * (max_ms_ - min_ms_));
  return true;
}

//------------------------------------------------------------------------------

Node::Node(uint32_t version, const std::string& name, const std::string& ip,
           uint16_t port, rpc::State state, const std::string& metadata)
    : version_(version),
      name_(name),
      addr_(ip, port),
      state_(state),
      metadata_(metadata) {}

Node::Node(const rpc::NodeMsg* msg)
    : version_(msg->version()),
      name_(msg->name()),
      addr_(msg->ip(), msg->port()),
      state_(msg->state()),
      metadata_(msg->metadata()) {}

Node& Node::operator=(const rpc::NodeMsg& msg) {
  if (name_ != msg.name() || msg.state() == rpc::State::UNKNOWN) {
    return *this;
  }

  version_ = msg.version();
  state_ = msg.state();
  // Force to cancel suspect timer.
  suspect_timer_ = nullptr;

  if (state_ == rpc::State::ALIVE) {
    addr_.set_ip(msg.ip());
    addr_.set_port(msg.port());
    metadata_ = msg.metadata();
  }

  return *this;
}

void Node::ToNodeMsg(const std::string& from, rpc::State state,
                     rpc::NodeMsg* msg) const {
  msg->set_version(version_);
  msg->set_name(name_);
  msg->set_ip(ip());
  msg->set_port(port());
  msg->set_state(state == rpc::State::UNKNOWN ? state_ : state);
  msg->set_metadata(metadata_);
  msg->set_from(from);
}

rpc::NodeMsg Node::ToNodeMsg(const std::string& from, rpc::State state) const {
  rpc::NodeMsg msg;
  ToNodeMsg(from, state, &msg);
  return msg;
}

rpc::ForwardMsg Node::ToForwardMsg(const std::string& from,
                                   rpc::State state) const {
  rpc::ForwardMsg msg;
  msg.set_ip(ip());
  msg.set_port(port());
  ToNodeMsg(from, state, msg.mutable_node());
  return msg;
}

bool Node::Conflict(const rpc::NodeMsg* msg) const {
  return (name_ == msg->name()) &&
         (ip() != msg->ip() || port() != msg->port() ||
          metadata_ != msg->metadata());
}

bool Node::Reset(const rpc::NodeMsg* msg) const {
  // Here, we can only use properties to recognize the reset. However, of
  // course, sometimes the node will reset without any changes. In these cases,
  // reset node will refute others outdated message in the future. Even if the
  // delay is uncertain, synchronization will eventually occur.
  return msg->state() == rpc::State::ALIVE && state_ == rpc::State::DEAD &&
         Conflict(msg);
}

uint32_t Node::version() const { return version_; }

void Node::set_version(uint32_t version) { version_ = version; }

const std::string& Node::name() const { return name_; }

const std::string& Node::ip() const { return addr_.ip()->ip(); }

uint16_t Node::port() const { return addr_.port(); }

const net::Address& Node::addr() const { return addr_; }

rpc::State Node::state() const { return state_; }

const std::string& Node::metadata() const { return metadata_; }

SuspectTimer::Ptr Node::suspect_timer() { return suspect_timer_; }

void Node::set_suspect_timer(SuspectTimer::Ptr suspect_timer) {
  suspect_timer_ = suspect_timer;
}

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

bool operator>(const Node& node, const rpc::NodeMsg& msg) {
  return node.name() == msg.name() && node.version() > msg.version();
}

bool operator<=(const Node& node, const rpc::NodeMsg& msg) {
  return node.name() == msg.name() && node.version() <= msg.version();
}

bool operator==(const Node& node, const rpc::NodeMsg& msg) {
  return node.version() == msg.version() && node.name() == msg.name() &&
         node.ip() == msg.ip() && node.port() == msg.port() &&
         node.state() == msg.state() && node.metadata() == msg.metadata();
}

}  // namespace gossip
}  // namespace common
