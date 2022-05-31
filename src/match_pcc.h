#ifndef MATCH_PCC_H
#define MATCH_PCC_H

#include "match.h"

void pcc_init(matcher_iface* matcher);

void pcc_cpu(match_result_t* result, bmp_img* src, bmp_img* tmpl, int thrd_cnt);
void pcc_gpu(match_result_t* result, bmp_img* src, bmp_img* tmpl, int blk_size,
             int thrd_size);

#endif  // MATCH_PCC_H