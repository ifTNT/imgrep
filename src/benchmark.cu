#include "benchmark.hpp"

static cudaEvent_t start, stop;

/**
 * Begin time measurement
 */
extern "C" void benchmark_gpu_begin() {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
}

/**
 * End time measurement
 */
extern "C" double benchmark_gpu_end() {
  float elapsed;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed /= 1.0E3;

  return (double)elapsed;
}