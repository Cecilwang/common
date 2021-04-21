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

#ifndef COMMON_NET_NET_H_
#define COMMON_NET_NET_H_

#include <ifaddrs.h>

#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "common/util/macro.h"
#include "common/util/type.h"

namespace common {
namespace net {

class IP;

std::string GetHostname();
std::set<std::shared_ptr<IP>> GetPublicIPs();
bool IsIPv4(const char* ip);
bool IsIPv6(const char* ip);

enum class IPType {
  kIPv4 = 0,
  kIPv6 = 1,
  kUnknown = 2,
};

class IP {
 public:
  IP(const char* ip, IPType type);
  virtual ~IP() = default;

  const std::string& ip() const;
  IPType type() const;

  virtual bool contain(const IP* other) const = 0;

  virtual std::string ToString(bool verbose = false) const = 0;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const IP& self);

 protected:
  std::string ip_;
  IPType type_ = IPType::kUnknown;
};

class IPv4 : public IP {
 public:
  explicit IPv4(const char* ip, uint32_t n_mask = 32);
  ~IPv4() = default;

  void set_mask(uint32_t mask);

  bool contain(const IP* other) const override;

  uint32_t val() const { return val_; }
  uint32_t mask() const { return mask_; }
  uint32_t subnetwork() const { return subnetwork_; }

  std::string ToString(bool verbose = false) const override;

 private:
  uint32_t val_ = 0;
  uint32_t mask_ = 0xFFFFFFFF;
  uint32_t subnetwork_;
};

class IPv6 : public IP {
 public:
  explicit IPv6(const char* ip, uint32_t n_mask = 128);
  ~IPv6() = default;

  void set_mask(uint8_t* mask);

  bool contain(const IP* other) const override;

  const util::UInt128& val() const { return val_; }
  const util::UInt128& mask() const { return mask_; }
  const util::UInt128& subnetwork() const { return subnetwork_; }

  std::string ToString(bool verbose = false) const override;

 private:
  util::UInt128 val_;
  util::UInt128 mask_;
  util::UInt128 subnetwork_;
};

std::shared_ptr<IP> CreateIP(const char* ip);
std::shared_ptr<IP> CreateIP(const std::string& ip);
std::shared_ptr<IP> CreateIP(const char* ip, uint32_t n_mask);
std::shared_ptr<IP> CreateIP(const std::string& ip, uint32_t n_mask);
std::shared_ptr<IP> CreateIP(ifaddrs* ifa);

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
  std::shared_ptr<IP> ip_;
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
  std::set<std::shared_ptr<IP>> ips_;

  DISALLOW_COPY_AND_ASSIGN(Node);
};

}  // namespace net
}  // namespace common

#endif  // COMMON_NET_NET_H_
