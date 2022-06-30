extern "C" {
#include "benchmark.h"
}
#include "cudautil.h"
#include <stdio.h>

static cudaEvent_t start, stop;

// [TODO] Benchmark using CUDA API will return invalid handle

/**
 * Begin time measurement
 */
void benchmark_gpu_begin() {

  GPU_ERRCHK(cudaEventCreate(&start));
  GPU_ERRCHK(cudaEventCreate(&stop));

  GPU_ERRCHK(cudaEventRecord(start));
}

/**
 * End time measurement
 */
double benchmark_gpu_end() {
  float elapsed;
  cudaEventRecord(stop);
  GPU_ERRCHK(cudaEventSynchronize(stop));
  GPU_ERRCHK(cudaEventElapsedTime(&elapsed, start, stop));
  elapsed /= 1.0E3;

  return (double)elapsed;
}