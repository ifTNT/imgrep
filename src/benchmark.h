/**
 * benchmark.h - Benchmark the CPU execution time or the one of GPU
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

/**
 * benchmark_cpu_begin/end() - Benchmark the total CPU execution time
 */
void benchmark_begin();
double benchmark_end();

/**
 * benchmark_gpu_begin/end() - Benchmark the total GPU execution time
 */
void benchmark_gpu_begin();
double benchmark_gpu_end();

#endif  // BENCHMARK_H