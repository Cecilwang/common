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

  // Force to cancel suspect timer.
  suspect_timer_ = nullptr;

  return *this;
}

Node& Node::operator=(const rpc::SuspectMsg& suspect) {
  if (name_ != suspect.dst()) {
    return *this;
  }
  version_ = suspect.version();
  state_ = rpc::State::SUSPECT;
  return *this;
}

Node& Node::operator=(const rpc::DeadMsg& dead) {
  if (name_ != suspect.dst()) {
    return *this;
  }
  version_ = dead.version();
  state_ = rpc::State::DEAD;
  // Force to cancel suspect timer.
  suspect_timer_ = nullptr;
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

uint32_t Node::version() const { return version_; }

void Node::set_version(uint32_t version) { version_ = version; }

const std::string& Node::name() const { return name_; }

const std::string& Node::ip() const { return addr_.ip()->ip(); }

uint16_t Node::port() const { return addr_.port(); }

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

}  // namespace gossip
}  // namespace common
