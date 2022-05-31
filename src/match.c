#include "match.h"

#include <stdlib.h>
#include <time.h>

void match_get_pos_list(match_pos_t** pos_list, match_result_t* src) {
  *pos_list = (match_pos_t*)malloc(sizeof(match_pos_t) * src->cnt);
  int pos_i = 0;
  for (unsigned int y = 0; y < src->height; y++) {
    for (unsigned int x = 0; x < src->width; x++) {
      if (src->map[y][x] == 1) {
        *pos_list[pos_i++] = (match_pos_t){x, y};
      }
    }
  }
}
