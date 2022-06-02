#include "match_ssd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "flatmat.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/**
 * _ssd_region - Calculate SSD of RGB channels within the given region.
 */
extern float _ssd_region(flatmat_t* src, flatmat_t* tmpl, int off_x, int off_y);

void _ssd_thrd(flatmat_t result, flatmat_t src, flatmat_t tmpl,
               match_work_region_t task) {
  uint off_x = task.off_x;
  uint off_y = task.off_y;
  for (int y = off_y; y < off_y + task.height && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.width && x < task.bound_x; x++) {
      *FLATMAT_AT(&result, x, y, 0) = _ssd_region(&src, &tmpl, x, y);
    }
  }
}

void ssd_init(matcher_iface* matcher) {
  matcher->match_cpu = &ssd_cpu;
  matcher->match_gpu = &ssd_gpu;
}

void ssd_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int thrd_cnt) {
  match_general_cpu(result, src, tmpl, thrd_cnt, &_ssd_thrd);
}