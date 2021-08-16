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

#include "common/cc/util/random.h"

namespace common {
namespace util {

RNG::RNG(int seed) : generator_(seed) {}

void RNG::SetSeed(int seed) { generator_.seed(seed); }

size_t RNG::Uniform(size_t low, size_t high) {
  std::uniform_int_distribution<size_t> distribution(low, high);
  return distribution(generator_);
}

size_t Uniform(size_t low, size_t high) { return kRNG.Uniform(low, high); }

}  // namespace util
}  // namespace common
