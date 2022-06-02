#include "match_pcc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "flatmat.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/**
 * _pcc_region - Calculate PCC of RGB channels within the given region
 */
extern float _pcc_region(flatmat_t* src, flatmat_t* tmpl, int off_x, int off_y);

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
      *FLATMAT_AT(&pcc_map, x, y, 0) = _pcc_region(src, tmpl, x, y);
      if (*FLATMAT_AT(&pcc_map, x, y, 0) > max)
        max = *FLATMAT_AT(&pcc_map, x, y, 0);
    }
  }
  printf("Max: %f\n", max);

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = (*FLATMAT_AT(&pcc_map, x, y, 0) == max);
      *FLATMAT_AT(&result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }
  flatmat_free(&pcc_map);
}