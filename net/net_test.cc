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

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"

#include "net/net.h"

namespace net {

TEST(TestIP, TestSimple) {
  EXPECT_TRUE(IsIPv4("0.0.0.0"));
  EXPECT_TRUE(IsIPv4("127.0.0.1"));
  EXPECT_FALSE(IsIPv4("300.0.0.1"));
  EXPECT_FALSE(IsIPv4("a.0.0.1"));
  EXPECT_FALSE(IsIPv4("-1.0.0.1"));

  EXPECT_TRUE(IsIPv6("::"));
  EXPECT_TRUE(IsIPv6("::1"));

  IP ip1("127.0.0.1");
  IP ip2(std::string("127.0.0.1"));
}

TEST(TestAddress, TestSimple) {
  Address a1("0.0.0.0", 0);
  Address a2("0.0.0.0", 0);
  EXPECT_EQ(a1, a2);
  Address a3("0.0.0.0", 1);
  EXPECT_NE(a1, a3);
  EXPECT_EQ(a1.ToString(), "0.0.0.0:0");
  EXPECT_EQ(std::string(a1), "0.0.0.0:0");
  EXPECT_EQ(static_cast<std::string>(a1), "0.0.0.0:0");
  std::cout << a1 << std::endl;
  std::ostringstream ss;
  ss << a1;
  EXPECT_EQ(ss.str(), "0.0.0.0:0");
}

TEST(TestNode, TestSimple) {
  Node n;
  std::cout << n << std::endl;
}

}  // namespace net
