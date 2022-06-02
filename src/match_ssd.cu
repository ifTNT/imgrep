extern "C" {
#include "match_ssd.h"
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
 * _ssd_region - Calculate SSD of RGB channels within the given region
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
                                   flatmat_t tmpl, task_t task) {
  uint off_x = (blockIdx.x * blockDim.x + threadIdx.x) * task.w;
  uint off_y = (blockIdx.y * blockDim.y + threadIdx.y) * task.h;
  for (int y = off_y; y < off_y + task.h && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.w && x < task.bound_x; x++) {
      *FLATMAT_AT(&result, x, y, 0) = _ssd_region(&src, &tmpl, x, y);
    }
  }
}

void ssd_gpu(match_result_t *result, flatmat_t *src, flatmat_t *tmpl,
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
  flatmat_t device_src, device_tmpl, device_ssd, device_result;
  flatmat_t host_ssd;

  flatmat_init(&host_ssd, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);
  flatmat_init_cuda(&device_ssd, result->width, result->height, 1);
  flatmat_init_cuda(&device_result, result->width, result->height, 1);
  flatmat_init_cuda(&device_src, src->width, src->height, src->layer);
  flatmat_init_cuda(&device_tmpl, tmpl->width, tmpl->height, tmpl->layer);

  // Copy data to device
  GPU_ERRCHK(
      cudaMemcpy(device_src.el, src->el, src->size, cudaMemcpyHostToDevice));
  GPU_ERRCHK(
      cudaMemcpy(device_tmpl.el, tmpl->el, tmpl->size, cudaMemcpyHostToDevice));

  // Calculate SSD map
  _ssd_kernel<<<blk_size_2d, thrd_size_2d>>>(device_ssd, device_src,
                                             device_tmpl, task_param);
  // Copy back the SSD map
  GPU_ERRCHK(cudaMemcpy(host_ssd.el, device_ssd.el, device_ssd.size,
                        cudaMemcpyDeviceToHost));

  GPU_ERRCHK(cudaPeekAtLastError());
  GPU_ERRCHK(cudaDeviceSynchronize());
  float max = -1.0E9;
  for (int y = 0; y < host_ssd.height; y++) {
    for (int x = 0; x < host_ssd.width; x++) {
      if (*FLATMAT_AT(&host_ssd, x, y, 0) > max)
        max = *FLATMAT_AT(&host_ssd, x, y, 0);
    }
  }

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = *FLATMAT_AT(&host_ssd, x, y, 0) == max;
      *FLATMAT_AT(&result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }

  // Cleanup
  flatmat_free_cuda(&device_src);
  flatmat_free_cuda(&device_tmpl);
  flatmat_free_cuda(&device_ssd);
  flatmat_free_cuda(&device_result);
  flatmat_free(&host_ssd);
}