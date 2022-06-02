#include "match_pcc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "flatmat.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/**
 * _pcc_region - Calculate PCC of RGB channels within the given region.
 */
extern float _pcc_region(flatmat_t* src, flatmat_t* tmpl, int off_x, int off_y);

void _pcc_thrd(flatmat_t result, flatmat_t src, flatmat_t tmpl,
               match_work_region_t task) {
  uint off_x = task.off_x;
  uint off_y = task.off_y;
  for (int y = off_y; y < off_y + task.height && y < task.bound_y; y++) {
    for (int x = off_x; x < off_x + task.width && x < task.bound_x; x++) {
      *FLATMAT_AT(&result, x, y, 0) = _pcc_region(&src, &tmpl, x, y);
    }
  }
}

void pcc_init(matcher_iface* matcher) {
  matcher->match_cpu = &pcc_cpu;
  matcher->match_gpu = &pcc_gpu;
}

void pcc_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int thrd_cnt) {
  match_general_cpu(result, src, tmpl, thrd_cnt, &_pcc_thrd);
}