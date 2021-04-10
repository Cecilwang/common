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

#include "gossip/gossip.h"

namespace gossip {

Cluster::Cluster(uint16_t port) : address_("0.0.0.0", port) {}

std::string Cluster::ToString() const {
  std::ostringstream ss;
  ss << "Cluster(self:" << self_ << " is listening on " << address_ << ")";
  return ss.str();
}

Cluster::operator std::string() const { return ToString(); }

std::ostream& operator<<(std::ostream& os, const Cluster& self) {
  return os << self.ToString();
}

}  // namespace gossip
