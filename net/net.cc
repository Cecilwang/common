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

#include <arpa/inet.h>
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

std::set<IP> GetIPs() {
  std::set<IP> ips;

  ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    LOG(WARNING) << "Failed to getifaddrs, errno: " << errno;
    return ips;
  }

  char ip[NI_MAXHOST];
  for (ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    auto addr = ifa->ifa_addr;
    if (addr != nullptr) {
      if (addr->sa_family == AF_INET) {
        if (getnameinfo(addr, sizeof(sockaddr_in), ip, sizeof(ip), nullptr, 0,
                        NI_NUMERICHOST) == 0 &&
            !common::StartsWith(ip, "127.")) {
          ips.insert(IP(ip));
        }
      } else if (addr->sa_family == AF_INET6) {
        if (getnameinfo(addr, sizeof(sockaddr_in6), ip, sizeof(ip), nullptr, 0,
                        NI_NUMERICHOST) == 0 &&
            strcmp(ip, "::1") != 0) {
          char* end = strchr(ip, '%');
          size_t len = end ? end - ip : strlen(ip);
          ips.insert(IP(std::string(ip, len)));
        }
      }
    }
  }

  freeifaddrs(ifaddr);

  return ips;
}

bool IsIPv4(const char* ip) {
  sockaddr_in sa;
  return inet_pton(AF_INET, ip, &(sa.sin_addr)) == 1;
}

bool IsIPv6(const char* ip) {
  sockaddr_in sa;
  return inet_pton(AF_INET6, ip, &(sa.sin_addr)) == 1;
}

//------------------------------------------------------------------------------

IP::IP(const char* ip) {
  if (IsIPv4(ip)) {
    ip_ = ip;
    type_ = IPType::kIPv4;
  } else if (IsIPv6(ip)) {
    ip_ = ip;
    type_ = IPType::kIPv6;
  }
}

IP::IP(const std::string& ip) : IP(ip.c_str()) {}

const std::string& IP::ip() const { return ip_; }

bool IP::operator<(const IP& other) const { return ip_ < other.ip_; }

bool IP::operator==(const IP& other) const { return ip_ == other.ip_; }

bool IP::operator!=(const IP& other) const { return !(*this == other); }

//------------------------------------------------------------------------------

Address::Address(const std::string& ip, uint16_t port) : ip_(ip), port_(port) {}

const std::string& Address::ip() const { return ip_.ip(); }
uint16_t Address::port() const { return port_; }

bool Address::operator==(const Address& other) const {
  return port_ == other.port_ && ip() == other.ip();
}

bool Address::operator!=(const Address& other) const {
  return !(*this == other);
}

std::string Address::ToString() const {
  std::ostringstream ss;
  ss << ip() << ":" << std::to_string(port_);
  return ss.str();
}

Address::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Address& self) {
  return os << self.ToString();
}

//------------------------------------------------------------------------------

Node::Node() : hostname_(GetHostname()), ips_(GetIPs()) {}

std::string Node::ToString() const {
  std::ostringstream ss;
  ss << "Node(" << hostname_ << ":[";
  bool first = true;
  for (const auto& x : ips_) {
    first ? ss << x.ip() : ss << ", " << x.ip();
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
