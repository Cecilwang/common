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

#include "glog/logging.h"

#include "common/util/time.h"

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
  timer_->Run();
}

bool SuspectTimer::AddSuspector(const std::string& suspector) {
  if (suspectors_.size() >= n_ ||
      suspectors_.find(suspector) != suspectors_.end()) {
    return false;
  }
  suspectors_.insert(suspector);
  size_t m = suspectors_.size();
  if (m >= n_) {
    timer_->set_timeout_ms(min_ms_);
  } else {
    timer_->set_timeout_ms(max_ms_ - log(m) / log(n_) * (max_ms_ - min_ms_));
  }
  return true;
}

std::string SuspectTimer::ToString() const {
  std::ostringstream ss;
  ss << "SuspectTimer: " << suspectors_.size() << "->" << n_ << ", " << *timer_;
  return ss.str();
}

SuspectTimer::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const SuspectTimer& self) {
  return os << self.ToString();
}

//------------------------------------------------------------------------------

Node::Node(const std::string& name, uint32_t version, const std::string& ip,
           uint16_t port, rpc::State state, const std::string& metadata)
    : name_(name),
      version_(version),
      addr_(ip, port),
      state_(state),
      metadata_(metadata),
      timestamp_ms_(util::NowInMS()) {
  VLOG(5) << ToString(true) << " was created.";
}

Node::Node(const rpc::NodeMsg* msg)
    : name_(msg->name()),
      version_(msg->version()),
      addr_(msg->ip(), msg->port()),
      state_(msg->state()),
      metadata_(msg->metadata()),
      timestamp_ms_(util::NowInMS()) {
  VLOG(5) << ToString(true) << " was created by " << msg->ShortDebugString();
}

Node& Node::operator=(const rpc::NodeMsg& msg) {
  if (name_ != msg.name() || msg.state() == rpc::State::UNKNOWN) {
    LOG(WARNING) << msg.ShortDebugString() << " failed to assign to " << *this;
    return *this;
  }
  VLOG(5) << ToString(true) << " = " << msg.ShortDebugString();

  version_ = msg.version();
  if (state_ == rpc::State::ALIVE) {
    addr_.set_ip(msg.ip());
    addr_.set_port(msg.port());
    metadata_ = msg.metadata();
  }
  if (state_ != msg.state()) {
    timestamp_ms_ = util::NowInMS();
  }
  if (suspect_timer_ && state_ != rpc::State::SUSPECT ||
      msg.state() != rpc::State::SUSPECT) {
    // Force to cancel suspect timer.
    suspect_timer_ = nullptr;
    LOG(WARNING) << *this
                 << "'s suspect timer has been canceled due to receiving "
                 << rpc::State_Name(msg.state());
  }
  state_ = msg.state();

  return *this;
}

void Node::ToNodeMsg(rpc::NodeMsg* msg) const {
  msg->set_name(name_);
  msg->set_version(version_);
  msg->set_ip(ip());
  msg->set_port(port());
  msg->set_state(state_);
  msg->set_metadata(metadata_);
}

bool Node::operator>(const rpc::NodeMsg& msg) {
  return name_ == msg.name() && version_ > msg.version();
}

bool Node::operator>=(const rpc::NodeMsg& msg) {
  return name_ == msg.name() && version_ >= msg.version();
}

bool Node::operator<(const rpc::NodeMsg& msg) {
  return name_ == msg.name() && version_ < msg.version();
}

bool Node::operator<=(const rpc::NodeMsg& msg) {
  return name_ == msg.name() && version_ <= msg.version();
}

bool Node::operator==(const rpc::NodeMsg& msg) {
  return name_ == msg.name() &&                       //
         version_ == msg.version() &&                 //
         ip() == msg.ip() && port() == msg.port() &&  //
         state_ == msg.state() &&                     //
         metadata_ == msg.metadata();
}

bool Node::operator!=(const rpc::NodeMsg& msg) { return !(*this == msg); }

bool Node::Reset(const rpc::NodeMsg* msg) const {
  // Here, we can only use properties to recognize the reset. However, of
  // course, sometimes the node will reset without any changes. In these cases,
  // reset node will refute others outdated message in the future. Even if the
  // delay is uncertain, synchronization will eventually occur.

  // TODO(sxwang): It is simpler to recognize by the timestamp_ms_.
  return name_ == msg->name() &&                                             //
         version_ >= msg->version() &&                                       //
         state_ == rpc::State::DEAD && msg->state() == rpc::State::ALIVE &&  //
         (ip() != msg->ip() ||                                               //
          port() != msg->port() ||                                           //
          metadata_ != msg->metadata());
}

const std::string& Node::name() const { return name_; }

uint32_t Node::version() const { return version_; }

void Node::set_version(uint32_t version) {
  LOG(INFO) << *this << " updated version: " << version_ << "->" << version;
  version_ = version;
}

const std::string& Node::ip() const { return addr_.ip()->ip(); }

uint16_t Node::port() const { return addr_.port(); }

const net::Address& Node::addr() const { return addr_; }

rpc::State Node::state() const { return state_; }

const std::string& Node::metadata() const { return metadata_; }

uint64_t Node::timestamp_ms() const { return timestamp_ms_; }

uint64_t Node::elapsed_ms() const { return util::NowInMS() - timestamp_ms_; }

SuspectTimer::Ptr Node::suspect_timer() { return suspect_timer_; }

void Node::set_suspect_timer(SuspectTimer::Ptr suspect_timer) {
  suspect_timer_ = suspect_timer;
  LOG(INFO) << *this << " set suspect_timer: " << *suspect_timer;
}

std::string Node::ToString(bool verbose) const {
  std::ostringstream ss;
  if (verbose) {
    ss << "Node(" << name_ << ", " << version_ << ", " << addr_ << ", "
       << rpc::State_Name(state_) << ", "
       << (state_ == rpc::State::SUSPECT && suspect_timer_
               ? suspect_timer_->ToString() + ", "
               : "")
       << timestamp_ms_ << (metadata_.empty() ? "" : ", " + metadata_) << ")";
  } else {
    ss << "Node(" << name_ << ")";
  }
  return ss.str();
}

Node::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Node& self) {
  return os << self.ToString();
}

}  // namespace gossip
}  // namespace common
