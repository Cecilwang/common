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

#ifndef COMMON_GOSSIP_NODE_H_
#define COMMON_GOSSIP_NODE_H_

#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>

#include "common/net/net.h"
#include "common/util/macro.h"
#include "common/util/thread.h"

#include "common/gossip/proto/gossip.pb.h"

namespace common {
namespace gossip {

struct Health {
  int score();

  Health& operator+=(int val);
  int operator*(int val);

 private:
  std::mutex mutex_;
  int score_ = 1;
  int upper_ = 8;
};

class SuspectTimer {
 public:
  typedef std::shared_ptr<SuspectTimer> Ptr;

  SuspectTimer(std::unique_ptr<util::Timer> timer, size_t n, uint64_t min_ms,
               uint64_t max_ms, const std::string& suspector);
  bool AddSuspector(const std::string& suspector);

 private:
  std::unique_ptr<util::Timer> timer_ = nullptr;
  size_t n_;
  uint64_t min_ms_;
  uint64_t max_ms_;
  std::unordered_set<std::string> suspectors_;

  DISALLOW_COPY_AND_ASSIGN(SuspectTimer);
};

class Node {
 public:
  typedef std::shared_ptr<Node> Ptr;
  typedef const std::shared_ptr<Node>& ConstPtr;

  Node(uint32_t version, const std::string& name, const std::string& ip,
       uint16_t port, rpc::State state, const std::string& metadata);
  explicit Node(const rpc::AliveMsg* alive);

  Node& operator=(const rpc::AliveMsg& alive);
  Node& operator=(const rpc::SuspectMsg& suspect);
  Node& operator=(const rpc::DeadMsg& dead);

  bool Conflict(const rpc::AliveMsg* alive) const;
  bool Reset(const rpc::AliveMsg* alive) const;

  uint32_t version() const;
  void set_version(uint32_t version);
  const std::string& name() const;
  const std::string& ip() const;
  uint16_t port() const;
  rpc::State state() const;
  const std::string& metadata() const;
  SuspectTimer::Ptr suspect_timer();
  void set_suspect_timer(SuspectTimer::Ptr suspect_timer);

  std::string ToString(bool verbose = false) const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Node& self);

 private:
  uint32_t version_;
  std::string name_;
  net::Address addr_;
  rpc::State state_;
  std::string metadata_;

  std::shared_ptr<SuspectTimer> suspect_timer_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(Node);
};

bool operator>(const Node& self, const rpc::AliveMsg& alive);
bool operator<=(const Node& self, const rpc::AliveMsg& alive);
bool operator==(const Node& self, const rpc::AliveMsg& alive);

}  // namespace gossip
}  // namespace common

#endif  // COMMON_GOSSIP_NODE_H_
