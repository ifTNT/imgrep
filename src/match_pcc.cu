extern "C" {
#include "match_pcc.h"
}
#include "flatmat.h"

#define CHANNELS (3)

typedef unsigned int uint;

typedef struct {
  uint w;
  uint h;
  uint bound_x;
  uint bound_y;
} task_t;

__device__ static float _region_pcc(flatmat_t *src, flatmat_t *tmpl, int off_x,
                                    int off_y) {
  int h = tmpl->height;
  int w = tmpl->width;
  int n = h * w;
  float pcc = 0;
  float p, q, sum_p, sum_q, sum_pp, sum_qq, sum_pq;
  for (int ch = 0; ch < src->layer; ch++) {
    sum_pq = 0, sum_p = 0, sum_q = 0, sum_pp = 0, sum_qq = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        p = *FLATMAT_GET(*src, x + off_x, y + off_y, ch);
        q = *FLATMAT_GET(*tmpl, x, y, ch);
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

__global__ static void _kernel_pcc(flatmat_t pcc, flatmat_t src, flatmat_t tmpl,
                                   task_t task) {
  uint off_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint off_y = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = off_y; y < off_y + task.h && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.w && x < task.bound_x; x++) {
      *FLATMAT_GET(pcc, x, y, 0) = _region_pcc(&src, &tmpl, x, y);
    }
  }
}

static void _malloc_host_device(void **host, void **device, uint n) {
  *host = malloc(n);
  cudaMalloc(device, n);
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
  // float *device_pcc, *host_pcc;
  // unsigned char *device_src, *host_src;
  // unsigned char *device_tmpl, *host_tmpl;
  // char *device_result, *host_result;
  // uint n_pcc = sizeof(float) * result->height * result->width;
  // uint n_result = sizeof(char) * result->height * result->width;
  // uint n_src = src->size;
  // uint n_tmpl = tmpl->size;
  flatmat_t device_src, device_tmpl, device_pcc, device_result;
  flatmat_t host_pcc;

  // _malloc_host_device((void **)&host_src, (void **)&device_src, n_src);
  // _malloc_host_device((void **)&host_tmpl, (void **)&device_tmpl,
  // n_tmpl); _malloc_host_device((void **)&host_pcc, (void **)&device_pcc,
  // n_pcc); _malloc_host_device((void **)&host_result, (void
  // **)&device_result, n_result);

  flatmat_init(&host_pcc, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);
  flatmat_init_cuda(&device_pcc, result->width, result->height, 1);
  flatmat_init_cuda(&device_result, result->width, result->height, 1);
  flatmat_init_cuda(&device_src, src->width, src->height, src->layer);
  flatmat_init_cuda(&device_tmpl, src->width, src->height, src->layer);

  // Copy data to device
  cudaMemcpy(device_src.el, src->el, src->size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_tmpl.el, tmpl->el, tmpl->size, cudaMemcpyHostToDevice);

  // Calculate PCC map
  _kernel_pcc<<<blk_size_2d, thrd_size_2d>>>(device_pcc, device_src,
                                             device_tmpl, task_param);
  // Copy back the PCC map
  cudaMemcpy(host_pcc.el, device_pcc.el, device_pcc.size,
             cudaMemcpyDeviceToHost);

  float max = -1.0;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      printf("%f ", *FLATMAT_GET(host_pcc, x, y, 0));
      if (*FLATMAT_GET(host_pcc, x, y, 0) > max)
        max = *FLATMAT_GET(host_pcc, x, y, 0);
    }
    printf("\n");
  }
  if (max > 3.0)
    max = 3.0;

  // Draw the result map
  // int tmp;
  // for (int y = 0; y < result->height; y++) {
  //   for (int x = 0; x < result->width; x++) {
  //     tmp = pcc_map[y][x] == max;
  //     result->map[y][x] = tmp;
  //     result->cnt += tmp;
  //   }
  // }

  // Cleanup
  flatmat_free_cuda(&device_src);
  flatmat_free_cuda(&device_tmpl);
  flatmat_free_cuda(&device_pcc);
  flatmat_free_cuda(&device_result);
  flatmat_free(&host_pcc);
}