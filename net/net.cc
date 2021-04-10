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

#include "net/net.h"

#include <ifaddrs.h>
#include <netdb.h>  // used by NI_MAXHOST
#include <unistd.h>

#include "glog/logging.h"

#include "common/string.h"

namespace net {

std::string GetHostname() {
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    return std::string(hostname);
  } else {
    LOG(WARNING) << "Failed to gethostname, errno: " << errno;
  }
  return "localhost";
}

std::set<std::string> GetIPv4s() {
  std::set<std::string> ipv4s;

  ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    LOG(WARNING) << "Failed to getifaddrs, errno: " << errno;
    return ipv4s;
  }

  char ipv4[NI_MAXHOST];
  for (ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    auto addr = ifa->ifa_addr;
    // clang-format off
    if (addr == nullptr ||
        addr->sa_family != AF_INET ||
        getnameinfo(addr, sizeof(sockaddr_in), ipv4, sizeof(ipv4), nullptr, 0, NI_NUMERICHOST) != 0 ||  // NOLINT
        common::StartsWith(ipv4, "127.")) {
      continue;
    }
    // clang-format on
    ipv4s.insert(std::string(ipv4));
  }

  freeifaddrs(ifaddr);

  return ipv4s;
}

//------------------------------------------------------------------------------

Address::Address(const std::string& ip, uint16_t port) : ip_(ip), port_(port) {}

const std::string& Address::ip() const { return ip_; }
uint16_t Address::port() const { return port_; }

bool Address::operator==(const Address& other) const {
  return port_ == other.port_ && ip_ == other.ip_;
}

bool Address::operator!=(const Address& other) const {
  return !(*this == other);
}

std::string Address::ToString() const {
  std::ostringstream ss;
  ss << ip_ << ":" << std::to_string(port_);
  return ss.str();
}

Address::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Address& self) {
  return os << self.ToString();
}

//------------------------------------------------------------------------------

Node::Node() : hostname_(GetHostname()), ips_(GetIPv4s()) {}

std::string Node::ToString() const {
  std::ostringstream ss;
  ss << "Node(" << hostname_ << ":[";
  bool first = true;
  for (const auto& x : ips_) {
    first ? ss << x : ss << ", " << x;
    first = false;
  }
  ss << "])";
  return ss.str();
}

Node::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Node& self) {
  return os << self.ToString();
}

}  // namespace net
