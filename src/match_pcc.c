#include "match_pcc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * _regoin_pcc - Calculate PCC of RGB channels within the given region
 */
#define CHANNELS (3)
#define PIXEL(img, x, y, ch) \
  ((float)((unsigned char*)(&img->img_pixels[(y)][(x)]))[ch])
static float _regoin_pcc(bmp_img* src, bmp_img* tmpl, int off_x, int off_y) {
  int h = tmpl->img_header.biHeight;
  int w = tmpl->img_header.biWidth;
  float mean_src, mean_tmpl;
  float cov, var_src, var_tmpl;
  float diff_src, diff_tmpl;
  float pcc = 0;
  for (int ch = 0; ch < CHANNELS; ch++) {
    // Calculate the mean value
    mean_src = 0, mean_tmpl = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        mean_src += PIXEL(src, x + off_x, y + off_y, ch);
        mean_tmpl += PIXEL(tmpl, x, y, ch);
      }
    }
    mean_src /= w * h;
    mean_tmpl /= w * h;

    // Calculate PCC
    cov = 0, var_src = 0, var_tmpl = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        diff_src = (PIXEL(src, x + off_x, y + off_y, ch) - mean_src);
        diff_tmpl = (PIXEL(tmpl, x, y, ch) - mean_tmpl);
        cov += diff_src * diff_tmpl;
        var_src += diff_src * diff_src;
        var_tmpl += diff_tmpl * diff_tmpl;
      }
    }
    pcc += cov / (sqrt(var_src) * sqrt(var_tmpl));
  }
  return pcc;
}
#undef TMPL_PIXEL
#undef SRC_PIXEL
#undef CHANNELS

void pcc_init(matcher_iface* matcher) {
  matcher->match_cpu = &pcc_cpu;
  matcher->match_gpu = &pcc_gpu;
}

void pcc_cpu(match_result_t* result, bmp_img* src, bmp_img* tmpl,
             int thrd_cnt) {
  float** pcc_map;

  // Initialize the result object and PCC map
  result->cnt = 0;
  result->height = src->img_header.biHeight - tmpl->img_header.biHeight + 1;
  result->width = src->img_header.biWidth - tmpl->img_header.biWidth + 1;
  result->map = (char**)malloc(sizeof(char*) * result->height);
  pcc_map = (float**)malloc(sizeof(float*) * result->height);
  for (int i = 0; i < result->height; i++) {
    result->map[i] = (char*)malloc(sizeof(char*) * result->height);
    pcc_map[i] = (float*)malloc(sizeof(float*) * result->height);
  }

  // Calculate PCC map
  float max = -1.0;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      pcc_map[y][x] = _regoin_pcc(src, tmpl, x, y);
      if (pcc_map[y][x] > max) max = pcc_map[y][x];
    }
  }
  if (max > 3.0) max = 3.0;

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = pcc_map[y][x] == max;
      result->map[y][x] = tmp;
      result->cnt += tmp;
    }
  }
}