#include "match_pcc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "flatmat.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/**
 * _region_pcc - Calculate PCC of RGB channels within the given region
 */
static float _region_pcc(flatmat_t* src, flatmat_t* tmpl, int off_x,
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

void pcc_init(matcher_iface* matcher) {
  matcher->match_cpu = &pcc_cpu;
  matcher->match_gpu = &pcc_gpu;
}

void pcc_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int thrd_cnt) {
  flatmat_t pcc_map;

  // Initialize the result object and PCC map
  result->cnt = 0;
  result->height = src->height - tmpl->height + 1;
  result->width = src->width - tmpl->width + 1;
  flatmat_init(&pcc_map, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);

  // Calculate PCC map
  float max = -1.0;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      *FLATMAT_GET(pcc_map, x, y, 0) = _region_pcc(src, tmpl, x, y);
      printf("%f ", *FLATMAT_GET(pcc_map, x, y, 0));
      if (*FLATMAT_GET(pcc_map, x, y, 0) > max)
        max = *FLATMAT_GET(pcc_map, x, y, 0);
    }
    printf("\n");
  }
  if (max > 3.0) max = 3.0;

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = (*FLATMAT_GET(pcc_map, x, y, 0) == max);
      *FLATMAT_GET(result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }
  flatmat_free(&pcc_map);
}