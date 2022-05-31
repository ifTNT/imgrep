#include "match_pcc.h"

void pcc_init(matcher_iface* matcher) {
  matcher->match_cpu = &pcc_cpu;
  matcher->match_gpu = &pcc_gpu;
}

void pcc_cpu(match_result_t* result, bmp_img* src, bmp_img* tmpl,
             int thrd_cnt) {}