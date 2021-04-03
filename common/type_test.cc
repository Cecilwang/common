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

#include "gtest/gtest.h"

#include "common/type.h"

namespace {
int& foo_lref();
int&& foo_rref();
int foo_value();
}  // namespace

namespace common {

TEST(TestTypename, TestAll) {
  int i = 0;
  const int ci = 0;

  EXPECT_EQ(Typename<decltype(i)>(), "i");
  EXPECT_EQ(Typename<decltype((i))>(), "i&");
  EXPECT_EQ(Typename<decltype(ci)>(), "i const");
  EXPECT_EQ(Typename<decltype((ci))>(), "i const&");
  EXPECT_EQ(Typename<decltype(static_cast<int&>(i))>(), "i&");
  EXPECT_EQ(Typename<decltype(static_cast<int&&>(i))>(), "i&&");
  EXPECT_EQ(Typename<decltype(static_cast<int>(i))>(), "i");
  EXPECT_EQ(Typename<decltype(foo_lref())>(), "i&");
  EXPECT_EQ(Typename<decltype(foo_rref())>(), "i&&");
  EXPECT_EQ(Typename<decltype(foo_value())>(), "i");
}

}  // namespace common
