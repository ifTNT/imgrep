extern "C" {
#include "match_ssd.h"
}
#include "flatmat.h"

typedef unsigned int uint;

/**
 * _ssd_region - Calculate SSD of RGB channels within the given region.
 */
extern "C" __host__ __device__ float
_ssd_region(flatmat_t *src, flatmat_t *tmpl, int off_x, int off_y) {
  int h = tmpl->height;
  int w = tmpl->width;
  float ssd = 0;
  float p, q, sum, diff;
  for (int ch = 0; ch < src->layer; ch++) {
    sum = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        p = *FLATMAT_AT(src, x + off_x, y + off_y, ch);
        q = *FLATMAT_AT(tmpl, x, y, ch);
        diff = p - q;
        sum += diff * diff;
      }
    }
    ssd += sum;
  }
  return -ssd;
}

/**
 * _ssd_kernel - The kernel to calculate SSD within given working area.
 */
__global__ static void _ssd_kernel(flatmat_t result, flatmat_t src,
                                   flatmat_t tmpl, match_work_region_t task) {
  uint off_x = (blockIdx.x * blockDim.x + threadIdx.x) * task.width;
  uint off_y = (blockIdx.y * blockDim.y + threadIdx.y) * task.height;
  for (int y = off_y; y < off_y + task.height && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.width && x < task.bound_x; x++) {
      *FLATMAT_AT(&result, x, y, 0) = _ssd_region(&src, &tmpl, x, y);
    }
  }
}

void ssd_gpu(match_result_t *result, flatmat_t *src, flatmat_t *tmpl,
             int blk_size, int thrd_size) {

  match_general_gpu(result, src, tmpl, blk_size, thrd_size, &_ssd_kernel);
}