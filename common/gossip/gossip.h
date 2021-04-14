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

#ifndef COMMON_GOSSIP_GOSSIP_H_
#define COMMON_GOSSIP_GOSSIP_H_

#include <string>

#include "common/net/net.h"
#include "common/util/macro.h"

namespace common {
namespace gossip {

class Cluster {
 public:
  explicit Cluster(uint16_t port = 2333);

  std::string ToString() const;
  operator std::string() const;
  friend std::ostream& operator<<(std::ostream& os, const Cluster& self);

 private:
  net::Node self_;
  net::Address address_;

  DISALLOW_COPY_AND_ASSIGN(Cluster);
};

}  // namespace gossip
}  // namespace common

#endif  // COMMON_GOSSIP_GOSSIP_H_
