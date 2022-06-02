extern "C" {
#include "match_pcc.h"
}
#include "cudautil.h"
#include "flatmat.h"

#define CHANNELS (3)

typedef unsigned int uint;

typedef struct {
  uint w;
  uint h;
  uint bound_x;
  uint bound_y;
} task_t;

/**
 * _pcc_region - Calculate PCC of RGB channels within the given region
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
  return pcc;
}

/**
 * _pcc_kernel - The kernel to calculate PCC within given working area.
 */
__global__ static void _pcc_kernel(flatmat_t pcc, flatmat_t src, flatmat_t tmpl,
                                   task_t task) {
  uint off_x = (blockIdx.x * blockDim.x + threadIdx.x) * task.w;
  uint off_y = (blockIdx.y * blockDim.y + threadIdx.y) * task.h;
  for (int y = off_y; y < off_y + task.h && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.w && x < task.bound_x; x++) {
      *FLATMAT_AT(&pcc, x, y, 0) = _pcc_region(&src, &tmpl, x, y);
    }
  }
}

void pcc_gpu(match_result_t *result, flatmat_t *src, flatmat_t *tmpl,
             int blk_size, int thrd_size) {

  // Initialize task parameter
  result->cnt = 0;
  result->height = src->height - tmpl->height + 1;
  result->width = src->width - tmpl->width + 1;
  uint thrd_cnt = blk_size * thrd_size;
  uint task_w = ceil((float)result->width / thrd_cnt);
  uint task_h = ceil((float)result->height / thrd_cnt);
  dim3 blk_size_2d(blk_size, blk_size);
  dim3 thrd_size_2d(thrd_size, thrd_size);
  task_t task_param{.w = task_w,
                    .h = task_h,
                    .bound_x = result->width,
                    .bound_y = result->height};

  // Initialize device and host resource
  flatmat_t device_src, device_tmpl, device_pcc, device_result;
  flatmat_t host_pcc;

  flatmat_init(&host_pcc, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);
  flatmat_init_cuda(&device_pcc, result->width, result->height, 1);
  flatmat_init_cuda(&device_result, result->width, result->height, 1);
  flatmat_init_cuda(&device_src, src->width, src->height, src->layer);
  flatmat_init_cuda(&device_tmpl, tmpl->width, tmpl->height, tmpl->layer);

  // Copy data to device
  GPU_ERRCHK(
      cudaMemcpy(device_src.el, src->el, src->size, cudaMemcpyHostToDevice));
  GPU_ERRCHK(
      cudaMemcpy(device_tmpl.el, tmpl->el, tmpl->size, cudaMemcpyHostToDevice));

  // Calculate PCC map
  _pcc_kernel<<<blk_size_2d, thrd_size_2d>>>(device_pcc, device_src,
                                             device_tmpl, task_param);
  // Copy back the PCC map
  GPU_ERRCHK(cudaMemcpy(host_pcc.el, device_pcc.el, device_pcc.size,
                        cudaMemcpyDeviceToHost));

  GPU_ERRCHK(cudaPeekAtLastError());
  GPU_ERRCHK(cudaDeviceSynchronize());
  float max = -1.0;
  for (int y = 0; y < host_pcc.height; y++) {
    for (int x = 0; x < host_pcc.width; x++) {
      if (*FLATMAT_AT(&host_pcc, x, y, 0) > max)
        max = *FLATMAT_AT(&host_pcc, x, y, 0);
    }
  }
  if (max > 3.0)
    max = 3.0;

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = *FLATMAT_AT(&host_pcc, x, y, 0) == max;
      *FLATMAT_AT(&result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }

  // Cleanup
  flatmat_free_cuda(&device_src);
  flatmat_free_cuda(&device_tmpl);
  flatmat_free_cuda(&device_pcc);
  flatmat_free_cuda(&device_result);
  flatmat_free(&host_pcc);
}