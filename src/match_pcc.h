#ifndef MATCH_PCC_H
#define MATCH_PCC_H

#include "flatmat.h"
#include "match.h"

void pcc_init(matcher_iface* matcher);

void pcc_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int thrd_cnt);
void pcc_gpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
             int blk_size, int thrd_size);

#endif  // MATCH_PCC_H