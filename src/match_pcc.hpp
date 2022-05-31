#ifndef MATCH_PCC_HPP
#define MATCH_PCC_HPP

#include "match.h"

extern "C" void pcc_gpu(match_result_t* result, bmp_img* src, bmp_img* tmpl,
                        int blk_size, int thrd_size);

#endif  // MATCH_PCC_HPP