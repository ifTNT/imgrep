#ifndef MATCH_H
#define MATCH_H

/**
 * match.h - The interface of different matcher.
 */

#include <libbmp/libbmp.h>

#include "flatmat.h"

/**
 * match_pos_t - The structure to store positions of result list.
 */
typedef struct {
  unsigned int x;
  unsigned int y;
} match_pos_t;

/**
 * match_work_region_t - The structure to pass the working region parameter to
 * each thread.
 *
 * off_x, off_y: Offset X and Y of this working region.
 * width, height: The width and height of this working region.
 * bound_x, bound_y: The boundary of total working region.
 */
typedef struct {
  unsigned int off_x;
  unsigned int off_y;
  unsigned int width;
  unsigned int height;
  unsigned int bound_x;
  unsigned int bound_y;
} match_work_region_t;

/**
 * match_result_t - The result from the matcher.
 *
 * cnt: How many target was founded.
 * map: The result map of founded target.
 *      1 means the upper left corner of the founded bounding box;
 *      0 otherwise.
 * elapsed: Elapsed time of calculation in second.
 */
typedef struct {
  int cnt;
  unsigned int width;
  unsigned int height;
  flatmat_t map;
} match_result_t;

/**
 * matcher_iface - The common public interface across different matcher
 *
 * result: The place where result should be store.
 * src: The object of the source image.
 * tmpl: The object of the template image.
 * thrd_cnt: (CPU mode only) The total worker thread count.
 * blk_size: (GPU mode only) The total block of CUDA.
 * thrd_size: (GPU mode only) How many threads per block of CUDA.
 */
typedef struct {
  void (*match_cpu)(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
                    int thrd_cnt);
  void (*match_gpu)(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
                    int blk_size, int thrd_size);
} matcher_iface;

/**
 * match_thrd_t - The unified interface of matcher's threads.
 * Each thread will calculate the similarity within the working region.
 * [Caution] All parameters use pass-by-value to achieve heterogeneous
 * programming with GPU.
 *
 * result: The place where similarity will store.
 * src: The source matrix to match.
 * tmpl: The template matrix to match.
 * task: The working region of this task.
 */
typedef void(match_thrd_t)(flatmat_t result, flatmat_t src, flatmat_t tmpl,
                           match_work_region_t task);

/**
 * match_get_pos_list - Retrieve the position list of founded result.
 *
 * pos_list: The position list to write.
 * src: The original result structure returned from the matcher.
 */
void match_get_pos_list(match_pos_t** pos_list, match_result_t* src);

/**
 * match_free_result - Release the resource hold by the result object.
 *
 * target: Target result to be release.
 */
void match_free_result(match_result_t* target);

/**
 * match_free_pos_list - Release the resource hold by the position list object.
 *
 * pos_list: Target list to be release.
 */
void match_free_pos_list(match_pos_t* pos_list);

/**
 * match_general_cpu, match_general_gpu - The common procedure across different
 * matchers.
 *
 * result: The place where similarity will store.
 * src: The source matrix to match.
 * tmpl: The template matrix to match.
 * thrd_cnt: (CPU mode only) The total worker thread count.
 * blk_size: (GPU mode only) The total block of CUDA.
 * thrd_size: (GPU mode only) How many threads per block of CUDA.
 * region_calc: The thread function to calculate similarity within a region
 */
void match_general_cpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
                       int thrd_cnt, match_thrd_t* region_calc);
void match_general_gpu(match_result_t* result, flatmat_t* src, flatmat_t* tmpl,
                       int blk_size, int thrd_size, match_thrd_t* region_calc);

#endif  // MATCH_H