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

#ifndef GOSSIP_GOSSIP_H_
#define GOSSIP_GOSSIP_H_

#include "common/macro.h"

namespace gossip {

class Cluster {
 public:
  Cluster() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(Cluster);
};

}  // namespace gossip

#endif  // GOSSIP_GOSSIP_H_
