#include "match.h"

#include <stdlib.h>
#include <time.h>

void match_get_pos_list(match_pos_t** pos_list, match_result_t* src) {
  *pos_list = (match_pos_t*)malloc(sizeof(match_pos_t) * src->cnt);
  int pos_i = 0;
  for (unsigned int y = 0; y < src->height; y++) {
    for (unsigned int x = 0; x < src->width; x++) {
      if (*FLATMAT_AT(&src->map, x, y, 0) == 1) {
        (*pos_list)[pos_i++] = (match_pos_t){x, y};
      }
    }
  }
}

void match_free_result(match_result_t* target) { flatmat_free(&target->map); }

void match_free_pos_list(match_pos_t* pos_list) { free(pos_list); }

void match_general_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
                       int thrd_cnt, match_thrd_t* region_calc) {
  flatmat_t similarly;

  // Initialize the result object and PCC map
  result->cnt = 0;
  result->height = src->height - tmpl->height + 1;
  result->width = src->width - tmpl->width + 1;
  flatmat_init(&similarly, result->width, result->height, 1);
  flatmat_init(&result->map, result->width, result->height, 1);

  // [TODO] Multithread using Pthread
  region_calc(similarly, *src, *tmpl,
              (match_work_region_t){.off_x = 0,
                                    .off_y = 0,
                                    .bound_x = result->width,
                                    .bound_y = result->height,
                                    .width = result->width,
                                    .height = result->height});

  // Find the maximal
  float max = -1.0;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      if (*FLATMAT_AT(&similarly, x, y, 0) > max)
        max = *FLATMAT_AT(&similarly, x, y, 0);
    }
  }

  // Draw the result map
  int tmp;
  for (int y = 0; y < result->height; y++) {
    for (int x = 0; x < result->width; x++) {
      tmp = (*FLATMAT_AT(&similarly, x, y, 0) == max);
      *FLATMAT_AT(&result->map, x, y, 0) = tmp;
      result->cnt += tmp;
    }
  }
  flatmat_free(&similarly);
}