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

#ifndef COMMON_UTIL_TYPE_H_
#define COMMON_UTIL_TYPE_H_

#include <cstdint>
#include <string>
#include <type_traits>

namespace common {
namespace util {

template <class T>
std::string Typename() {
  typedef typename std::remove_reference<T>::type NT;
  std::string r = typeid(NT).name();
  if (std::is_const<NT>::value) r += " const";
  if (std::is_volatile<NT>::value) r += " volatile";
  if (std::is_lvalue_reference<T>::value) r += "&";
  if (std::is_rvalue_reference<T>::value) r += "&&";
  return r;
}

struct UInt128 {
  UInt128() = default;
  UInt128(uint64_t high, uint64_t low);
  explicit UInt128(const uint8_t u8[16]);

  void set1(uint32_t n = 128);

  UInt128 operator|(const UInt128& other);
  UInt128 operator&(const UInt128& other);
  UInt128 operator~(void);
  UInt128 operator^(const UInt128& other);

  bool operator<(const UInt128& other) const;
  bool operator<=(const UInt128& other) const;
  bool operator>(const UInt128& other) const;
  bool operator>=(const UInt128& other) const;
  bool operator==(const UInt128& other) const;
  bool operator!=(const UInt128& other) const;

  uint64_t high = 0ull;
  uint64_t low = 0ull;
};

}  // namespace util
}  // namespace common

#endif  // COMMON_UTIL_TYPE_H_
