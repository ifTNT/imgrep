extern "C" {
#include "match_pcc.h"
}
#include "flatmat.h"

typedef unsigned int uint;

/**
 * _pcc_region - Calculate PCC of RGB channels within the given region.
 */
extern "C" __host__ __device__ float
_pcc_region(flatmat_t *src, flatmat_t *tmpl, int off_x, int off_y) {
  int h = tmpl->height;
  int w = tmpl->width;
  int n = h * w;
  float pcc = 0;
  float p, q, sum_p, sum_q, sum_pp, sum_qq, sum_pq;
  for (int ch = 0; ch < src->layer; ch++) {
    sum_pq = 0, sum_p = 0, sum_q = 0, sum_pp = 0, sum_qq = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        p = *FLATMAT_AT(src, x + off_x, y + off_y, ch);
        q = *FLATMAT_AT(tmpl, x, y, ch);
        sum_p += p;
        sum_q += q;
        sum_pp += p * p;
        sum_qq += q * q;
        sum_pq += p * q;
      }
    }
    pcc += (sum_pq * n - sum_p * sum_q) / sqrt(sum_qq * n - sum_q * sum_q) /
           sqrt(sum_pp * n - sum_p * sum_p);
  }
  return isfinite(pcc) ? pcc : 0;
}

/**
 * _pcc_kernel - The kernel to calculate PCC within given working area.
 */
__global__ static void _pcc_kernel(flatmat_t result, flatmat_t src,
                                   flatmat_t tmpl, match_work_region_t task) {
  uint off_x = (blockIdx.x * blockDim.x + threadIdx.x) * task.width;
  uint off_y = (blockIdx.y * blockDim.y + threadIdx.y) * task.height;
  for (int y = off_y; y < off_y + task.height && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.width && x < task.bound_x; x++) {
      *FLATMAT_AT(&result, x, y, 0) = _pcc_region(&src, &tmpl, x, y);
    }
  }
}

void pcc_gpu(match_result_t *result, flatmat_t *src, flatmat_t *tmpl,
             int blk_size, int thrd_size) {
  match_general_gpu(result, src, tmpl, blk_size, thrd_size, &_pcc_kernel);
}