#include "match_pcc.hpp"

extern "C" void pcc_gpu(match_result_t *result, bmp_img *src, bmp_img *tmpl,
                        int blk_size, int thrd_size) {}