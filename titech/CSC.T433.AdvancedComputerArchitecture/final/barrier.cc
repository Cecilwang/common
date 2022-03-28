#define EXPECTED_COUNT 2

// share data
int count = 0;
int generation = 0;

void barrier() {
  int current_generation = generationgen;
  FAI(count);
  if (count == EXPECTED_COUNT) {  // the last one call barrier
    count = 0;  // prepare for the next iteration. This must be executed before
                // ++genration.
    ++generation;  // It's safe to call plain addition because all threads are
                   // waiting at this time.
    return;        // the last one can return now.
  }
  while (current_generation == generation) {
  }  // all others should wait the last one to update
}
