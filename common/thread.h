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

#ifndef COMMON_THREAD_H_
#define COMMON_THREAD_H_

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <utility>

#include "common/macro.h"

namespace common {

class Thread {
 public:
  Thread() = default;
  ~Thread();

  void Idle(uint64_t ms);
  void Stop();

  void set_running(bool running);
  void WaitUntilStop();

 protected:
  bool running_ = false;
  std::thread thread_;
  std::condition_variable cv_;
  std::mutex mutex_;

  DISALLOW_COPY_AND_ASSIGN(Thread);
};

template <class F>
class ThreadWrap : public Thread {
 public:
  explicit ThreadWrap(F&& f) : f_(std::move(f)) {}

  void Run() {
    if (!running_) {
      set_running(true);
      thread_ = std::thread([this] { f_(this); });
    }
  }

 private:
  F f_;
  DISALLOW_COPY_AND_ASSIGN(ThreadWrap);
};

template <class F>
std::unique_ptr<ThreadWrap<F>> CreateThread(F&& f) {
  return std::unique_ptr<ThreadWrap<F>>(new ThreadWrap<F>(std::forward<F>(f)));
}

template <class F>
class LoopThreadWrap : public Thread {
 public:
  LoopThreadWrap(F&& f, uint64_t interval_ms)
      : f_(std::move(f)), interval_ms_(interval_ms) {}

  bool running() {
    std::unique_lock<std::mutex> lock(mutex_);
    return !cv_.wait_for(lock, interval_ms_, [this] { return !running_; });
  }

  void Run() {
    if (!running_) {
      set_running(true);
      thread_ = std::thread([this] {
        for (; running();) {
          f_();
        }
      });
    }
  }

 private:
  F f_;
  std::chrono::milliseconds interval_ms_;
  DISALLOW_COPY_AND_ASSIGN(LoopThreadWrap);
};

template <class F>
std::unique_ptr<LoopThreadWrap<F>> CreateLoopThread(F&& f,
                                                    uint64_t interval_ms) {
  return std::unique_ptr<LoopThreadWrap<F>>(
      new LoopThreadWrap<F>(std::forward<F>(f), interval_ms));
}

}  // namespace common

#endif  // COMMON_THREAD_H_
