extern "C" {
#include "match.h"
}
#include "cudautil.h"

extern "C" void match_general_gpu(match_result_t *result, flatmat_t *src,
                                  flatmat_t *tmpl, int blk_size, int thrd_size,
                                  match_thrd_t *region_calc) {
  // Initialize task parameter
  result->cnt = 0;
  result->height = src->height - tmpl->height + 1;
  result->width = src->width - tmpl->width + 1;
  uint thrd_cnt = blk_size * thrd_size;
  uint task_w = ceil((float)result->width / thrd_cnt);
  uint task_h = ceil((float)result->height / thrd_cnt);
  dim3 blk_size_2d(blk_size, blk_size);
  dim3 thrd_size_2d(thrd_size, thrd_size);
  match_work_region_t task_param{.off_x = 0, // Unused
                                 .off_y = 0, // Unused
                                 .width = task_w,
                                 .height = task_h,
                                 .bound_x = result->width,
                                 .bound_y = result->height};

  // Initialize device and host resource
  flatmat_t device_src, device_tmpl, device_similarly, device_result;
  flatmat_t host_similarly;

  flatmat_init(&host_similarly, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);
  flatmat_init_cuda(&device_similarly, result->width, result->height, 1);
  flatmat_init_cuda(&device_result, result->width, result->height, 1);
  flatmat_init_cuda(&device_src, src->width, src->height, src->layer);
  flatmat_init_cuda(&device_tmpl, tmpl->width, tmpl->height, tmpl->layer);

  // Copy data to device
  GPU_ERRCHK(
      cudaMemcpy(device_src.el, src->el, src->size, cudaMemcpyHostToDevice));
  GPU_ERRCHK(
      cudaMemcpy(device_tmpl.el, tmpl->el, tmpl->size, cudaMemcpyHostToDevice));

  // Calculate similarly map
  region_calc<<<blk_size_2d, thrd_size_2d>>>(device_similarly, device_src,
                                             device_tmpl, task_param);
  // Copy back the similarly map
  GPU_ERRCHK(cudaMemcpy(host_similarly.el, device_similarly.el,
                        device_similarly.size, cudaMemcpyDeviceToHost));

  GPU_ERRCHK(cudaPeekAtLastError());
  GPU_ERRCHK(cudaDeviceSynchronize());
  float max = -1.0E9;
  for (int y = 0; y < host_similarly.height; y++) {
    for (int x = 0; x < host_similarly.width; x++) {
      if (*FLATMAT_AT(&host_similarly, x, y, 0) > max)
        max = *FLATMAT_AT(&host_similarly, x, y, 0);
    }
  }

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = *FLATMAT_AT(&host_similarly, x, y, 0) == max;
      *FLATMAT_AT(&result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }

  // Cleanup
  flatmat_free_cuda(&device_src);
  flatmat_free_cuda(&device_tmpl);
  flatmat_free_cuda(&device_similarly);
  flatmat_free_cuda(&device_result);
  flatmat_free(&host_similarly);
}