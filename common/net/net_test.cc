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

#include "common/net/net.h"

namespace common {
namespace net {

TEST(TestIP, TestSimple) {
  EXPECT_TRUE(IsIPv4("0.0.0.0"));
  EXPECT_TRUE(IsIPv4("127.0.0.1"));
  EXPECT_FALSE(IsIPv4("300.0.0.1"));
  EXPECT_FALSE(IsIPv4("a.0.0.1"));
  EXPECT_FALSE(IsIPv4("-1.0.0.1"));
  EXPECT_FALSE(IsIPv4("::"));
  EXPECT_FALSE(IsIPv4("::1"));

  EXPECT_TRUE(IsIPv6("::"));
  EXPECT_TRUE(IsIPv6("::1"));
  EXPECT_FALSE(IsIPv6("127.0.0.1"));

  {
    IPv4 ip("127.0.0.1", 0);
    EXPECT_EQ(ip.type(), IPType::kIPv4);
    EXPECT_EQ(ip.val(), 2130706433);
    EXPECT_EQ(ip.mask(), 0);
    EXPECT_EQ(ip.subnetwork(), 0);
  }
  {
    IPv4 ip("127.0.0.1", 8);
    EXPECT_EQ(ip.type(), IPType::kIPv4);
    EXPECT_EQ(ip.val(), 2130706433);
    EXPECT_EQ(ip.mask(), 0xFF000000);
    EXPECT_EQ(ip.subnetwork(), 2130706432);
  }
  {
    IPv4 ip("127.0.0.1");
    EXPECT_EQ(ip.type(), IPType::kIPv4);
    EXPECT_EQ(ip.val(), 2130706433);
    EXPECT_EQ(ip.mask(), 0xFFFFFFFF);
    EXPECT_EQ(ip.subnetwork(), 2130706433);
  }
  {
    IPv6 ip("2001:0db8:85a3:08d3:1319:8a2e:0370:7344", 0);
    EXPECT_EQ(ip.type(), IPType::kIPv6);
    EXPECT_EQ(ip.val().high, 0x20010db885a308d3);
    EXPECT_EQ(ip.val().low, 0x13198a2e03707344);
    EXPECT_EQ(ip.mask().high, 0);
    EXPECT_EQ(ip.mask().low, 0);
    EXPECT_EQ(ip.subnetwork().high, 0);
    EXPECT_EQ(ip.subnetwork().low, 0);
  }
  {
    IPv6 ip("2001:0db8:85a3:08d3:1319:8a2e:0370:7344", 64);
    EXPECT_EQ(ip.type(), IPType::kIPv6);
    EXPECT_EQ(ip.val().high, 0x20010db885a308d3);
    EXPECT_EQ(ip.val().low, 0x13198a2e03707344);
    EXPECT_EQ(ip.mask().high, 0xFFFFFFFFFFFFFFFF);
    EXPECT_EQ(ip.mask().low, 0);
    EXPECT_EQ(ip.subnetwork().high, 0x20010db885a308d3);
    EXPECT_EQ(ip.subnetwork().low, 0);
  }
  {
    IPv6 ip("2001:0db8:85a3:08d3:1319:8a2e:0370:7344");
    EXPECT_EQ(ip.type(), IPType::kIPv6);
    EXPECT_EQ(ip.val().high, 0x20010db885a308d3);
    EXPECT_EQ(ip.val().low, 0x13198a2e03707344);
    EXPECT_EQ(ip.mask().high, 0xFFFFFFFFFFFFFFFF);
    EXPECT_EQ(ip.mask().low, 0xFFFFFFFFFFFFFFFF);
    EXPECT_EQ(ip.subnetwork().high, 0x20010db885a308d3);
    EXPECT_EQ(ip.subnetwork().low, 0x13198a2e03707344);
  }
}

TEST(TestAddress, TestSimple) {
  Address a1("0.0.0.0", 0);
  Address a2("0.0.0.0", 0);
  EXPECT_EQ(a1, a2);
  Address a3("0.0.0.0", 1);
  EXPECT_NE(a1, a3);
  Address a4("0.0.0.1", 0);
  EXPECT_NE(a1, a4);
  EXPECT_EQ(a1.ToString(), "0.0.0.0:0");
  EXPECT_EQ(std::string(a1), "0.0.0.0:0");
  EXPECT_EQ(static_cast<std::string>(a1), "0.0.0.0:0");
  std::ostringstream ss;
  ss << a1;
  EXPECT_EQ(ss.str(), "0.0.0.0:0");
}

TEST(TestHostname, TestPrint) { std::cout << GetHostname() << std::endl; }

TEST(TestPublicIPs, TestPrint) {
  const auto& ips = GetPublicIPs();
  for (const auto& x : ips) {
    std::cout << x << std::endl;
  }
}

TEST(TestNode, TestDelegateIP) {
  auto ip1 = CreateIP("127.0.0.1");
  auto ip2 = CreateIP("2001:0db8:85a3:08d3:1319:8a2e:0370:7344");
  std::vector<std::shared_ptr<IP>> ips = {ip1, ip2};

  EXPECT_EQ(GetDelegateIP(ips, "127.0.0.2"), nullptr);
  EXPECT_EQ(GetDelegateIP(ips, ip1), ip1);
  EXPECT_EQ(GetDelegateIP(ips, "0.0.0.0"), ip1);
  EXPECT_EQ(GetDelegateIP(ips, ip2), ip2);
  EXPECT_EQ(GetDelegateIP(ips, "::"), ip2);
}

}  // namespace net
}  // namespace common
