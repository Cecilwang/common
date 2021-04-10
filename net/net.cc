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

namespace net {

Address::Address(const std::string& ip, uint8_t port) : ip_(ip), port_(port) {}

const std::string& Address::ip() const { return ip_; }
uint8_t Address::port() const { return port_; }

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

}  // namespace net
