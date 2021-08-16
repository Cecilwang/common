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

#ifndef COMMON_CC_UTIL_GENERATOR_H_
#define COMMON_CC_UTIL_GENERATOR_H_

#include <condition_variable>  // NOLINT
#include <exception>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "common/cc/util/macro.h"

namespace common {
namespace util {

template <class T>
class Generator {
 public:
  enum class State {
    kProduced = 0,  // waiting for consumption
    kConsumed = 1,  // waiting for production
    kFinished = 2,
  };

  class FinishedException : public std::exception {
   public:
    const char* what() const throw() override {
      return "Generator has finished.";
    }
  };

  class IteratorWrap;

  class Iterator {
   public:
    friend class IteratorWrap;

    Iterator() = default;
    ~Iterator() {
      if (thread_.joinable()) {
        thread_.join();
      }
    }

    void Start(const std::function<void(Generator<T>::Iterator*)>& f) {
      thread_ = std::thread([this, &f] {
        f(this);
        Finish();
      });
    }

    void Yield(T data) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == State::kConsumed; });
      promise_.set_value(data);
      state_ = State::kProduced;
    }

    std::future<T> Next() {
      std::lock_guard<std::mutex> guard(mutex_);
      promise_ = std::promise<T>();
      state_ = State::kConsumed;
      cv_.notify_one();
      return promise_.get_future();
    }

    void Finish() {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == State::kConsumed; });
      state_ = State::kFinished;
      try {
        throw FinishedException();
      } catch (...) {
        try {
          promise_.set_exception(std::current_exception());
        } catch (...) {
        }  // set_exception() may throw too
      }
    }

   private:
    T data_;
    std::thread thread_;
    std::promise<T> promise_;
    State state_ = State::kProduced;
    std::mutex mutex_;
    std::condition_variable cv_;

    DISALLOW_COPY_AND_ASSIGN(Iterator);
  };

  class IteratorWrap {
   public:
    friend class Generator;

    explicit IteratorWrap(Iterator* it = nullptr) : it_(it) {}

    IteratorWrap& operator++() {
      if (it_) {
        try {
          it_->data_ = it_->Next().get();
        } catch (const FinishedException&) {
          it_.reset(nullptr);
        }
      }
      return *this;
    }

    const T& operator*() const { return it_->data_; }
    const T* operator->() const { return &(it_->data_); }

    bool operator!=(const IteratorWrap& other) const {
      return it_ != other.it_;
    }

   private:
    std::unique_ptr<Iterator> it_ = nullptr;
  };

  explicit Generator(std::function<void(Generator<T>::Iterator*)>&& f)
      : f_(f) {}

  IteratorWrap begin() {
    IteratorWrap itw(new Iterator());
    itw.it_->Start(f_);
    ++itw;  // Get the first element;
    return itw;
  }
  IteratorWrap end() const { return IteratorWrap(); }

 private:
  std::function<void(Generator<T>::Iterator*)> f_;

  DISALLOW_COPY_AND_ASSIGN(Generator);
};

}  // namespace util
}  // namespace common

#endif  // COMMON_CC_UTIL_GENERATOR_H_
