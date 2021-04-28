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

#ifndef COMMON_UTIL_THREAD_H_
#define COMMON_UTIL_THREAD_H_

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <utility>

#include "common/util/macro.h"
#include "common/util/random.h"
#include "common/util/time.h"

namespace common {
namespace util {

class Thread {
 public:
  Thread() = default;
  ~Thread();

  virtual void Run() = 0;
  void Idle(uint64_t ms);
  void Stop();

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

  void Run() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      running_ = true;
      thread_ = std::thread([this] { f_(this); });
    }
  }

 private:
  F f_;
  DISALLOW_COPY_AND_ASSIGN(ThreadWrap);
};

template <class F>
std::unique_ptr<Thread> CreateThread(F&& f) {
  return std::unique_ptr<Thread>(new ThreadWrap<F>(std::forward<F>(f)));
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

  void Run() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      running_ = true;
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
std::unique_ptr<Thread> CreateLoopThread(F&& f, uint64_t interval_ms,
                                         bool delay = false) {
  if (delay) {
    SleepForMS(Uniform(0, interval_ms));
  }
  return std::unique_ptr<Thread>(
      new LoopThreadWrap<F>(std::forward<F>(f), interval_ms));
}

class Timer : public Thread {
 public:
  explicit Timer(uint64_t timeout_ms) : timeout_ms_(timeout_ms) {}

  void Run() override;
  virtual void _Run() = 0;

  void set_timeout_ms(uint64_t timeout_ms);

 protected:
  std::chrono::milliseconds timeout_ms_;

  bool breath_ = false;
  std::chrono::time_point<std::chrono::system_clock> start_;

  DISALLOW_COPY_AND_ASSIGN(Timer);
};

template <class F>
class TimerWrap : public Timer {
 public:
  TimerWrap(F&& f, uint64_t timeout_ms) : Timer(timeout_ms), f_(std::move(f)) {}

  void _Run() override { f_(); }

 private:
  F f_;
  DISALLOW_COPY_AND_ASSIGN(TimerWrap);
};

template <class F>
std::unique_ptr<Timer> CreateTimer(F&& f, uint64_t timeout_ms) {
  return std::unique_ptr<Timer>(
      new TimerWrap<F>(std::forward<F>(f), timeout_ms));
}

}  // namespace util
}  // namespace common

#endif  // COMMON_UTIL_THREAD_H_
