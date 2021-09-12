#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "common/cc/util/time.h"

size_t kCacheLine = 64;
size_t kL1 = 49152;
size_t kL2 = 524288;
size_t kL3 = 6291456;

inline void* memalign(size_t n_bytes) {
  void* ptr = nullptr;
  CHECK_EQ(posix_memalign(&ptr, kCacheLine, n_bytes), 0);
  return ptr;
}

template <class T>
class Vector {
 public:
  Vector(size_t n = 0)
      : n_(n), data_(static_cast<T*>(memalign(n * sizeof(T)))) {}
  ~Vector() { free(data_); }

  void set(size_t n) {
    n_ = n;
    if (data_ != nullptr) {
      free(data_);
      data_ = nullptr;
    }
    data_ = static_cast<T*>(memalign(n * sizeof(T)));
  }

  void Loop(size_t s) {
    Measure(
        [this, s] {
          for (size_t j = 0; j < n_; j += s) {
            ++data_[j];
          }
        },
        "loop stride " + std::to_string(s));
  }

  void InnerLoop(size_t n, size_t m) {
    Measure(
        [this, n, m] {
          for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
              ++data_[i * m + j];
            }
          }
        },
        "inner loop");
  }

  void OuterLoop(size_t n, size_t m) {
    Measure(
        [this, n, m] {
          for (size_t j = 0; j < m; ++j) {
            for (size_t i = 0; i < n; ++i) {
              ++data_[i * m + j];
            }
          }
        },
        "outer loop");
  }

  size_t size() const { return n_; }
  T* data() { return data_; }
  T& operator[](size_t i) { return data_[i]; }

 private:
  size_t n_ = 0;
  T* data_ = nullptr;
};

template <class F>
void Measure(F&& f, const std::string& name, int n = 10) {
  // flush
  Vector<int> a(kL3 * 2);
  for (size_t i = 0; i < a.size(); ++i) {
    ++a[i];
  }
  // f(); // warmup
  auto us = common::util::NowInUS();
  for (int i = 0; i < n; ++i) {
    f();
  }
  LOG(INFO) << name << " " << (common::util::NowInUS() - us) / n << "us.";
}

class TestLocality : public ::testing::Test {
 public:
};

TEST_F(TestLocality, TestLoop1) {
  Vector<int> v(1000000);
  int* vp = v.data();
  Measure(
      [&] {
        for (size_t i = 0; i < v.size(); ++i) {
          ++vp[0];
        }
      },
      "fixed");
  Measure(
      [&] {
        for (size_t i = 0; i < v.size(); ++i) {
          ++vp[i];
        }
      },
      "loop");
}

TEST_F(TestLocality, TestLoop2) {
  size_t n = 1024;
  size_t m = 1024;
  Vector<int> v(n * m);
  v.InnerLoop(n, m);
  v.OuterLoop(n, m);
}

TEST_F(TestLocality, TestStride) {
  Vector<int> v(kL1 * 1000);
  v.Loop(1);
  v.Loop(2);
  v.Loop(4);
  v.Loop(8);
  v.Loop(16);
  v.Loop(24);
  v.Loop(32);
}

TEST_F(TestLocality, TestFusion) {
  Vector<int> x(kL1 * 1000);
  Vector<int> y(kL1 * 1000);
  size_t n = x.size();
  int* xp = x.data();
  int* yp = y.data();
  Measure(
      [&] {
        for (size_t i = 0; i < n; ++i) {
          ++xp[i];
        }
        for (size_t i = 0; i < n; ++i) {
          yp[i] = xp[i] * 2;
        }
      },
      "separated");
  Measure(
      [&] {
        for (size_t i = 0; i < n; ++i) {
          ++xp[i];
          yp[i] = xp[i] * 2;
        }
      },
      "fused");
}

TEST_F(TestLocality, TestAlign) {
  std::vector<Vector<char>> a(1000000, Vector<char>(kCacheLine * 2));
  Measure(
      [&] {
        for (size_t i = 0; i < a.size(); ++i) {
          char* p = a[i].data();
          for (size_t j = 0; j < kCacheLine; ++j) {
            ++p[j];
          }
        }
      },
      "aligned");
  Measure(
      [&] {
        for (size_t i = 0; i < a.size(); ++i) {
          char* p = a[i].data();
          for (size_t j = 4; j < kCacheLine + 4; ++j) {
            ++p[j];
          }
        }
      },
      "unaligned");
}

TEST_F(TestLocality, TestMM) {
  size_t n = 1;
  size_t m = 128;
  size_t t = kL1 * 2;
  Vector<int> a(n * t);
  Vector<int> b(t * m);
  Vector<int> c(n * m);
  int* ap = a.data();
  int* bp = b.data();
  int* cp = c.data();

  Measure(
      [&] {
        size_t ao = 0;
        size_t co = 0;
        for (size_t i = 0; i < n; ++i) {
          for (size_t j = 0; j < m; ++j) {
            size_t bo = j;
            for (size_t k = 0; k < t; ++k) {
              cp[co + j] += ap[ao + k] * bp[bo];
              bo += m;
            }
          }
          ao += t;
          co += m;
        }
      },
      "Row*Col");

  Measure(
      [&] {
        size_t ao = 0;
        size_t co = 0;
        for (size_t i = 0; i < n; ++i) {
          size_t bo = 0;
          for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < t; ++k) {
              cp[co + j] += ap[ao + k] * bp[bo + k];
            }
            bo += t;
          }
          ao += t;
          co += m;
        }
      },
      "Row*Row");

  auto pack = [&](size_t g) {
    Measure(
        [&, t = t / g] {
          for (int q = 0; q < g; ++q) {
            size_t ao = 0;
            size_t co = 0;
            int* ap = a.data();
            int* bp = b.data();
            for (size_t i = 0; i < n; ++i) {
              size_t bo = 0;
              for (size_t j = 0; j < m; ++j) {
                for (size_t k = 0; k < t; ++k) {
                  cp[co + j] += ap[ao + k] * bp[bo + k];
                }
                bo += t;
              }
              ao += t;
              co += m;
            }
            ap += n * t;
            bp += m * t;
          }
        },
        "Pack " + std::to_string(g));
  };
  pack(2);
  pack(4);
  pack(8);
}
