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

#ifndef COMMON_UTIL_RANDOM_H_
#define COMMON_UTIL_RANDOM_H_

#include <random>

namespace common {
namespace util {

class RNG {
 public:
  explicit RNG(int seed);
  void SetSeed(int seed);
  size_t Uniform(size_t low, size_t high);

 private:
  std::default_random_engine generator_;
};

static RNG kRNG(0);
size_t Uniform(size_t low, size_t high);

}  // namespace util
}  // namespace common

#endif  // COMMON_UTIL_RANDOM_H_
