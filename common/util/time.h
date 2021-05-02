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

#ifndef COMMON_UTIL_TIME_H_
#define COMMON_UTIL_TIME_H_

#include "common/util/type.h"

#include <chrono>  // NOLINT

namespace common {
namespace util {

template <class T>
uint64_t TimePointToMS(const T& tp) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             tp.time_since_epoch())
      .count();
}

uint64_t NowInMS();
void SleepForMS(uint64_t ms);

}  // namespace util
}  // namespace common

#endif  // COMMON_UTIL_TIME_H_
