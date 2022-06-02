#ifndef MATCH_SSD_H
#define MATCH_SSD_H

#include "flatmat.h"
#include "match.h"

void ssd_init(matcher_iface* matcher);

void ssd_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int thrd_cnt);
void ssd_gpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int blk_size, int thrd_size);

#endif  // MATCH_SSD_H