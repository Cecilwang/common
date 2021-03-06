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

#include "common/cc/util/string.h"

#include <cstring>

namespace common {
namespace util {

bool StartsWith(const char* str, const char* pattern) {
  return strlen(pattern) <= strlen(str) &&
         strncmp(str, pattern, strlen(pattern)) == 0;
}

bool StartsWith(const std::string& str, const std::string& pattern) {
  return StartsWith(str.c_str(), pattern.c_str());
}

}  // namespace util
}  // namespace common
