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

Thread::~Thread() { Stop(); }

void Thread::Idle(uint64_t ms) { SleepForMS(ms); }

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

void Timer::Run() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    running_ = true;
    start_ = std::chrono::system_clock::now();
    thread_ = std::thread([this] {
      std::unique_lock<std::mutex> lock(mutex_);
      while (true) {
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
      _Run();
    });
  }
}

void Timer::set_timeout_ms(uint64_t timeout_ms) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    breath_ = true;
    timeout_ms_ = std::chrono::milliseconds(timeout_ms);
  }
  cv_.notify_all();
}

}  // namespace util
}  // namespace common
