#include "match_pcc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * _regoin_pcc - Calculate PCC of RGB channels within the given region
 */
#define PIXEL(img, x, y, ch) \
  ((float)((unsigned char*)(&(img)->img_pixels[(y)][(x)]))[(ch)])
static float _regoin_pcc(bmp_img* src, bmp_img* tmpl, int off_x, int off_y) {
  const int CHANNELS = 3;
  int h = tmpl->img_header.biHeight;
  int w = tmpl->img_header.biWidth;
  int n = h * w;
  float pcc = 0;
  float p, q, sum_p, sum_q, sum_pp, sum_qq, sum_pq;
  for (int ch = 0; ch < CHANNELS; ch++) {
    sum_pq = 0, sum_p = 0, sum_q = 0, sum_pp = 0, sum_qq = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        p = PIXEL(src, x + off_x, y + off_y, ch);
        q = PIXEL(tmpl, x, y, ch);
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
#undef PIXEL

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