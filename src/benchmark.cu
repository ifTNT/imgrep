extern "C" {
#include "benchmark.h"
}

static cudaEvent_t start, stop;

/**
 * Begin time measurement
 */
void benchmark_gpu_begin() {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
}

/**
 * End time measurement
 */
double benchmark_gpu_end() {
  float elapsed;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1.0E3;

  return (double)elapsed;
}