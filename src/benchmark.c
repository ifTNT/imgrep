#include "benchmark.h"

#include <stdlib.h>
#include <time.h>

static struct timespec start, finish;

/**
 * Begin wall-time measurement
 */
void benchmark_begin() { clock_gettime(CLOCK_MONOTONIC, &start); }

/**
 * End time measurement
 */
double benchmark_end() {
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1.0E9;
  return elapsed;
}