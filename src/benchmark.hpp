/**
 * benchmark.hpp - Benchmark the CPU execution time or the one of GPU (C++
 * interface)
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

/**
 * benchmark_gpu_begin/end() - Benchmark the total GPU execution time
 */
extern "C" void benchmark_gpu_begin();
extern "C" double benchmark_gpu_end();

#endif  // BENCHMARK_H