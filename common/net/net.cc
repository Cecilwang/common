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

#include "common/net/net.h"

#include <arpa/inet.h>
#include <netdb.h>  // used by NI_MAXHOST
#include <unistd.h>

#include <algorithm>
#include <vector>

#include "glog/logging.h"

#include "common/util/string.h"

namespace common {
namespace net {

IP::IP(const char* ip, IPType type) : ip_(ip), type_(type) {}

const std::string& IP::ip() const { return ip_; }

IPType IP::type() const { return type_; }

bool IP::operator==(const IP& other) const { return ip_ == other.ip_; }
bool IP::operator!=(const IP& other) const { return ip_ != other.ip_; }

IP::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const IP& self) {
  return os << self.ToString();
}

//------------------------------------------------------------------------------

IPv4::IPv4(const char* ip, uint32_t n_mask) : IP(ip, IPType::kIPv4) {
  in_addr addr;
  inet_pton(AF_INET, ip_.c_str(), &addr);
  val_ = ntohl(addr.s_addr);
  n_mask = std::min(n_mask, 32u);
  mask_ = n_mask ? ~((1u << (32 - n_mask)) - 1) : 0;
  subnetwork_ = val_ & mask_;
}

void IPv4::set_mask(uint32_t mask) {
  mask_ = mask;
  subnetwork_ = val_ & mask_;
}

bool IPv4::contain(const IP* other) const {
  if (type_ != other->type()) {
    return false;
  }
  const IPv4* ipv4 = dynamic_cast<const IPv4*>(other);
  return subnetwork_ == ipv4->subnetwork_;
}

std::string IPv4::ToString(bool verbose) const {
  std::ostringstream ss;
  ss << ip_;
  if (verbose) {
    ss << "[subnetwork: " << subnetwork_ << "]";
  }
  return ss.str();
}

//------------------------------------------------------------------------------

IPv6::IPv6(const char* ip, uint32_t n_mask) : IP(ip, IPType::kIPv6) {
  in6_addr addr;
  inet_pton(AF_INET6, ip_.c_str(), &addr);
  val_ = util::UInt128(addr.s6_addr);
  mask_.set1(n_mask);
  subnetwork_ = val_ & mask_;
}

void IPv6::set_mask(uint8_t* mask) {
  mask_ = util::UInt128(mask);
  subnetwork_ = val_ & mask_;
}

bool IPv6::contain(const IP* other) const {
  if (type_ != other->type()) {
    return false;
  }
  const IPv6* ipv6 = dynamic_cast<const IPv6*>(other);
  return subnetwork_ == ipv6->subnetwork_;
}

std::string IPv6::ToString(bool verbose) const {
  std::ostringstream ss;
  ss << ip_;
  if (verbose) {
    ss << "[subnetwork: " << subnetwork_.high << " " << subnetwork_.low << "]";
  }
  return ss.str();
}

//------------------------------------------------------------------------------

#define IFAIsIPv4(src) (src->ifa_addr->sa_family == AF_INET)

#define GetIPv4FromIFA(src, dest)                                      \
  (getnameinfo(src->ifa_addr, sizeof(sockaddr_in), dest, sizeof(dest), \
               nullptr, 0, NI_NUMERICHOST) == 0)

#define GetIPv4MaskFromIFA(src) \
  ntohl(reinterpret_cast<sockaddr_in*>(src->ifa_netmask)->sin_addr.s_addr)

#define IFAIsIPv6(src) (src->ifa_addr->sa_family == AF_INET6)

#define GetIPv6FromIFA(src, dest)                                       \
  (getnameinfo(src->ifa_addr, sizeof(sockaddr_in6), dest, sizeof(dest), \
               nullptr, 0, NI_NUMERICHOST) == 0)

#define GetIPv6MaskFromIFA(src) \
  (reinterpret_cast<sockaddr_in6*>(src->ifa_netmask)->sin6_addr.s6_addr)

const std::vector<IP::Ptr> kPrivateIPs({
    // RFC6890
    CreateIP("0.0.0.0", 8),
    CreateIP("10.0.0.0", 8),
    CreateIP("100.64.0.0", 10),
    CreateIP("127.0.0.0", 8),
    CreateIP("169.254.0.0", 16),
    CreateIP("172.16.0.0", 12),
    CreateIP("192.0.0.0", 24),
    CreateIP("192.0.0.0", 29),
    CreateIP("192.0.2.0", 24),
    CreateIP("192.88.99.0", 24),
    CreateIP("192.168.0.0", 16),
    CreateIP("198.18.0.0", 15),
    CreateIP("198.51.100.0", 24),
    CreateIP("203.0.113.0", 24),
    CreateIP("240.0.0.0", 4),
    CreateIP("255.255.255.255", 32),
    CreateIP("::1", 128),
    CreateIP("::", 128),
    CreateIP("64:ff9b::", 96),
    CreateIP("::ffff:0:0", 96),
    CreateIP("100::", 64),
    CreateIP("2001::", 16),
    CreateIP("2002::", 16),
    CreateIP("fc00::", 7),
    CreateIP("fe80::", 10),
    // Docker
    CreateIP("172.17.0.0", 16),
});

const char* GetHostname() {
  static char hostname[256] = "";
  if (strlen(hostname) == 0) {
    if (gethostname(hostname, sizeof(hostname)) != 0) {
      LOG(WARNING) << "Failed to gethostname, errno: " << errno;
      strncpy(hostname, "localhost", strlen("localhost") + 1);
    }
  }
  return hostname;
}

bool IsIPv4(const char* ip) {
  in_addr addr;
  return inet_pton(AF_INET, ip, &addr) == 1;
}

bool IsIPv6(const char* ip) {
  in6_addr addr;
  return inet_pton(AF_INET6, ip, &addr) == 1;
}

IP::Ptr CreateIP(const char* ip) {
  if (IsIPv4(ip)) {
    return IP::Ptr(new IPv4(ip));
  } else if (IsIPv6(ip)) {
    return IP::Ptr(new IPv6(ip));
  }
  return nullptr;
}

IP::Ptr CreateIP(const std::string& ip) { return CreateIP(ip.c_str()); }

IP::Ptr CreateIP(const char* ip, uint32_t n_mask) {
  if (IsIPv4(ip) && n_mask <= 32) {
    return IP::Ptr(new IPv4(ip, n_mask));
  } else if (IsIPv6(ip) && n_mask <= 128) {
    return IP::Ptr(new IPv6(ip, n_mask));
  }
  return nullptr;
}

IP::Ptr CreateIP(const std::string& ip, uint32_t n_mask) {
  return CreateIP(ip.c_str(), n_mask);
}

IP::Ptr CreateIP(ifaddrs* ifa) {
  if (ifa->ifa_addr == nullptr) {
    return nullptr;
  }
  char ip[NI_MAXHOST];
  if (IFAIsIPv4(ifa) && GetIPv4FromIFA(ifa, ip)) {
    auto ret = new IPv4(ip);
    if (ifa->ifa_netmask) {
      ret->set_mask(GetIPv4MaskFromIFA(ifa));
    }
    return IP::Ptr(ret);
  }
  if (IFAIsIPv6(ifa) && GetIPv6FromIFA(ifa, ip)) {
    if (char* end = strchr(ip, '%')) {
      *end = '\0';
    }
    auto ret = new IPv6(ip);
    if (ifa->ifa_netmask) {
      ret->set_mask(GetIPv6MaskFromIFA(ifa));
    }
    return IP::Ptr(ret);
  }
  return nullptr;
}

const std::vector<IP::Ptr>& GetPublicIPs() {
  static std::vector<IP::Ptr> ips;
  if (ips.empty()) {
    ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
      LOG(WARNING) << "Failed to getifaddrs, errno: " << errno;
      return ips;
    }

    for (ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (IP::Ptr ip = CreateIP(ifa)) {
        for (const auto& x : kPrivateIPs) {
          if (x->contain(ip.get())) {
            ip = nullptr;
            break;
          }
        }
        if (ip != nullptr) {
          ips.push_back(ip);
        }
      }
    }

    freeifaddrs(ifaddr);
  }
  return ips;
}

IP::Ptr GetDelegateIP(const std::vector<IP::Ptr>& ips, IP::Ptr ip) {
  bool any_ipv4 = ip->ip() == "0.0.0.0";
  bool any_ipv6 = ip->ip() == "::";
  for (const auto& x : ips) {
    if (ip->contain(x.get()) || (any_ipv4 && x->type() == IPType::kIPv4) ||
        (any_ipv6 && x->type() == IPType::kIPv6)) {
      return x;
    }
  }
  return nullptr;
}

IP::Ptr GetDelegateIP(const std::vector<IP::Ptr>& ips, const char* ip) {
  return GetDelegateIP(ips, CreateIP(ip));
}

//------------------------------------------------------------------------------

Address::Address(const std::string& ip, uint16_t port)
    : ip_(CreateIP(ip)), port_(port) {}

Address::Address(IP::ConstPtr ip, uint16_t port) : ip_(ip), port_(port) {}

void Address::set_ip(const std::string& ip) { ip_ = CreateIP(ip); }

IP::ConstPtr Address::ip() const { return ip_; }

void Address::set_port(uint16_t port) { port_ = port; }

uint16_t Address::port() const { return port_; }

bool Address::operator==(const Address& other) const {
  return port_ == other.port_ && ((*ip_) == (*(other.ip_)));
}

bool Address::operator!=(const Address& other) const {
  return !(*this == other);
}

std::string Address::ToString() const {
  std::ostringstream ss;
  ss << *ip_ << ":" << std::to_string(port_);
  return ss.str();
}

Address::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Address& self) {
  return os << self.ToString();
}

}  // namespace net
}  // namespace common
