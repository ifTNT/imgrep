#include "match_ssd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "flatmat.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/**
 * _ssd_region - Calculate SSD of RGB channels within the given region
 */
extern float _ssd_region(flatmat_t* src, flatmat_t* tmpl, int off_x, int off_y);

void ssd_init(matcher_iface* matcher) {
  matcher->match_cpu = &ssd_cpu;
  matcher->match_gpu = &ssd_gpu;
}

void ssd_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int thrd_cnt) {
  flatmat_t ssd_map;

  // Initialize the result object and SSD map
  result->cnt = 0;
  result->height = src->height - tmpl->height + 1;
  result->width = src->width - tmpl->width + 1;
  flatmat_init(&ssd_map, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);

  // Calculate SSD map
  float max = -1E9;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      *FLATMAT_AT(&ssd_map, x, y, 0) = _ssd_region(src, tmpl, x, y);
      if (*FLATMAT_AT(&ssd_map, x, y, 0) > max)
        max = *FLATMAT_AT(&ssd_map, x, y, 0);
    }
  }

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = (*FLATMAT_AT(&ssd_map, x, y, 0) == max);
      *FLATMAT_AT(&result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }
  flatmat_free(&ssd_map);
}