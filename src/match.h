#ifndef MATCH_H
#define MATCH_H

/**
 * match.h - The interface of different matcher.
 */

#include <libbmp/libbmp.h>

typedef struct {
  unsigned int x;
  unsigned int y;
} match_pos_t;

/**
 * match_result_t - The result from the matcher.
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
  char** map;
} match_result_t;

/**
 * matcher_iface - The common public interface across different matcher
 * result: The place where result should be store.
 * src: The object of the source image.
 * tmpl: The object of the template image.
 * thrd_cnt: (CPU mode only) The total worker thread count.
 * blk_size: (GPU mode only) The total block of CUDA
 * thrd_size: (GPU mode only) How many threads per block of CUDA
 */
typedef struct {
  void (*match_cpu)(match_result_t* result, bmp_img* src, bmp_img* tmpl,
                    int thrd_cnt);
  void (*match_gpu)(match_result_t* result, bmp_img* src, bmp_img* tmpl,
                    int blk_size, int thrd_size);
} matcher_iface;

/**
 * match_get_pos_list - Retrieve the position list of founded result.
 * pos_list: The position list to write.
 * src: The original result structure returned from the matcher.
 */
void match_get_pos_list(match_pos_t** pos_list, match_result_t* src);

/**
 * match_free_result - Release the resource hold by the result object.
 * target: Target result to be release.
 */
void match_free_result(match_result_t* target);

/**
 * match_free_pos_list - Release the resource hold by the position list object.
 * pos_list: Target list to be release.
 */
void match_free_pos_list(match_pos_t* pos_list);

#endif  // MATCH_H