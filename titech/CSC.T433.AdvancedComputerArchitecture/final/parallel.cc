#include <barrier>
#include <condition_variable>
#include <iostream>
#include <thread>

#define N 8      /* the number of grids */
#define NCORES 4 /* the number of cores */
#define TOL 15.0 /* tolerance parameter */

// share data
float A[N + 2], B[N + 2];
float diff = 0.0;

// lock and barrier
std::mutex mutex;
#define LOCK mutex.lock()
#define UNLOCK mutex.unlock()
std::barrier barrier(NCORES);
#define BARRIER barrier.arrive_and_wait()
//#define BARRIER

void solve_pp(int pid, int ncores) {
  int i, done = 0;                    /* private variables */
  int mymin = 1 + (pid * N / ncores); /* private variable */
  int mymax = mymin + N / ncores - 1; /* private variable */
  // DEBUG
  LOCK;
  std::cout << pid << " " << mymin << " " << mymax << std::endl;
  UNLOCK;

  // init
  for (i = mymin; i <= std::min(mymax, N - 2); i++) A[i] = B[i] = 100 + i * i;
  BARRIER;  // Make sure all threads are initialized.

  while (!done) {
    for (i = mymin; i <= mymax; i++) { /* use A as input */
      B[i] = 0.333 * (A[i - 1] + A[i] + A[i + 1]);
    }
    BARRIER;  // Next loop depends B across different threads

    float mydiff = 0;
    for (i = mymin; i <= mymax; i++) { /* use B as input */
      A[i] = 0.333 * (B[i - 1] + B[i] + B[i + 1]);
      mydiff += std::abs(B[i] - A[i]);
    }
    LOCK;  // Update diff by mydiff
    diff += mydiff;
    UNLOCK;
    BARRIER;  // Make sure all threads are updated
    if (diff < TOL) done = 1;
    BARRIER;  // Make sure all threads accessed diff before reset
    if (pid == 0) diff = 0;
    // clang-format off
    // BARRIER; // This BARRIER is unnecessary because the first BARRIER in the
                // while loop is guaranteed to be synchronized.
    // clang-format on
  }
}

int main() {
  std::thread threads[NCORES];
  for (int i = 0; i < NCORES; ++i) {
    threads[i] = std::thread([=] { solve_pp(i, NCORES); });
  }
  for (int i = 0; i < NCORES; ++i) {
    threads[i].join();
  }
}
