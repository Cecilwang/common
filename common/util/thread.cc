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

#include "common/util/thread.h"

namespace common {
namespace util {

void Thread::Run() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    running_ = true;
    RunThread();
  }
}

void Thread::Stop() {
  mutex_.lock();
  if (running_) {
    running_ = false;
    mutex_.unlock();
    cv_.notify_all();
    thread_.join();
  } else {
    mutex_.unlock();
  }
}

void Thread::WaitUntilStop() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !running_; });
}

bool Thread::running() {
  bool ret;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ret = running_;
  }
  cv_.notify_all();
  return ret;
}

std::condition_variable& Thread::cv() { return cv_; }

std::mutex& Thread::mutex() { return mutex_; }

//------------------------------------------------------------------------------

void Timer::RunThread() {
  start_ = std::chrono::system_clock::now();
  thread_ = std::thread([this] {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (cv_.wait_until(lock, start_ + timeout_ms_,
                         [this] { return !running_ || breath_; })) {
        if (!running_) {  // cancel
          return;
        } else {  // change timeout
          breath_ = false;
        }
      } else {  // timeout, call func
        break;
      }
    }
    CallBack();
  });
}

uint64_t Timer::timeout_ms() {
  uint16_t ret;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ret = timeout_ms_.count();
  }
  cv_.notify_all();
  return ret;
}

uint64_t Timer::end_ms() {
  uint16_t ret;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ret = util::TimePointToMS(start_ + timeout_ms_);
  }
  cv_.notify_all();
  return ret;
}

void Timer::set_timeout_ms(uint64_t timeout_ms) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    breath_ = true;
    timeout_ms_ = std::chrono::milliseconds(timeout_ms);
  }
  cv_.notify_all();
}

std::ostream& operator<<(std::ostream& os, Timer& self) {
  if (self.running()) {
    auto now = util::NowInMS();
    auto end = self.end_ms();
    os << now << "->" << end << ":";
    if (end <= now) {
      os << "end";
    } else {
      os << end - now << "ms";
    }
  } else {
    os << " has not started.";
  }
  return os;
}

}  // namespace util
}  // namespace common
