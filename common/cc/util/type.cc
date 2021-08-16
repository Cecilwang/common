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

#include "common/cc/util/type.h"

#include <algorithm>

namespace {
uint64_t UInt8ToUInt64(const uint8_t* u8) {
  return (static_cast<uint64_t>(u8[0]) << 56) |
         (static_cast<uint64_t>(u8[1]) << 48) |
         (static_cast<uint64_t>(u8[2]) << 40) |
         (static_cast<uint64_t>(u8[3]) << 32) |
         (static_cast<uint64_t>(u8[4]) << 24) |
         (static_cast<uint64_t>(u8[5]) << 16) |
         (static_cast<uint64_t>(u8[6]) << 8) |
         (static_cast<uint64_t>(u8[7]) << 0);
}
}  // namespace

namespace common {
namespace util {

UInt128::UInt128(const uint8_t u8[16]) {
  high = UInt8ToUInt64(u8);
  low = UInt8ToUInt64(u8 + 8);
}
UInt128::UInt128(uint64_t high, uint64_t low) : high(high), low(low) {}

void UInt128::set1(uint32_t n) {
  if (n == 0) {
    high = low = 0ull;
    return;
  }
  n = std::min(n, 128u);
  if (n <= 64) {
    high = ~((1ull << (64 - n)) - 1);
  } else {
    high = 0xFFFFFFFFFFFFFFFF;
    n -= 64;
    low = ~((1ull << (64 - n)) - 1);
  }
}

UInt128 UInt128::operator|(const UInt128& other) {
  return {high | other.high, low | other.low};
}
UInt128 UInt128::operator&(const UInt128& other) {
  return {high & other.high, low & other.low};
}
UInt128 UInt128::operator~(void) { return {~high, ~low}; }
UInt128 UInt128::operator^(const UInt128& other) {
  return {high ^ other.high, low ^ other.low};
}

bool UInt128::operator<(const UInt128& other) const {
  return high < other.high || (high == other.high && low < other.low);
}
bool UInt128::operator<=(const UInt128& other) const {
  return high < other.high || (high == other.high && low <= other.low);
}
bool UInt128::operator>(const UInt128& other) const {
  return high > other.high || (high == other.high && low > other.low);
}
bool UInt128::operator>=(const UInt128& other) const {
  return high > other.high || (high == other.high && low >= other.low);
}
bool UInt128::operator==(const UInt128& other) const {
  return high == other.high && low == other.low;
}
bool UInt128::operator!=(const UInt128& other) const {
  return !((*this) == other);
}

}  // namespace util
}  // namespace common
