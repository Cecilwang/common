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

#ifndef NET_NET_H_
#define NET_NET_H_

#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "common/macro.h"

namespace net {

std::string GetHostname();
std::set<std::string> GetIPv4s();

class Address {
 public:
  explicit Address(const std::string& ip = "0.0.0.0", uint16_t port = 80);

  const std::string& ip() const;
  uint16_t port() const;

  bool operator==(const Address& other) const;
  bool operator!=(const Address& other) const;

  std::string ToString() const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Address& self);

 private:
  std::string ip_;
  uint16_t port_;

  DISALLOW_COPY_AND_ASSIGN(Address);
};

class Node {
 public:
  Node();

  std::string ToString() const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Node& self);

 private:
  std::string hostname_;
  std::set<std::string> ips_;

  DISALLOW_COPY_AND_ASSIGN(Node);
};

}  // namespace net

#endif  // NET_NET_H_
