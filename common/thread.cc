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

#include "common/thread.h"

#include "common/time.h"

namespace common {

Thread::~Thread() { Stop(); }

void Thread::Idle(uint64_t ms) { SleepForMS(ms); }

void Thread::Stop() {
  if (running_) {
    set_running(false);
    cv_.notify_all();
    thread_.join();
  }
}

void Thread::set_running(bool running) {
  std::lock_guard<std::mutex> lock(mutex_);
  running_ = running;
}

void Thread::WaitUntilStop() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !running_; });
}

}  // namespace common
