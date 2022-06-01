#ifndef FLATMAT_H
#define FLATMAT_H

#include <libbmp/libbmp.h>

/**
 * flatmat_t - The structure of flatmatrix object
 */
typedef struct {
  float* el;            // The place to store element data
  unsigned int width;   // Width of this matrix
  unsigned int height;  // Height of this matrix
  unsigned int size;    // Bytes of whole element data
  unsigned int layer;   // How many layers are staked
} flatmat_t;

/**
 * flatmat_init(_cuda) - Initialize flatmatrix.
 */
void flatmat_init(flatmat_t* dst, unsigned int width, unsigned int height,
                  unsigned int layer);
void flatmat_init_cuda(flatmat_t* dst, unsigned int width, unsigned int height,
                       unsigned int layer);
/**
 * flatmat_from_bmp - Create flatmatrix object from BMP image.
 */
void flatmat_from_bmp(flatmat_t* dst, bmp_img src);

/**
 * FLATMAT_GET - Get element of one channel from specified matrix.
 */
#define FLATMAT_GET(mat, x, y, z) \
  (&(mat).el[(y) * (mat).width * (mat).layer + (x) * (mat).layer + (z)])

/**
 * flatmat_free(_cuda) - Release the resource hold by flatmatrix object
 */
void flatmat_free(flatmat_t* mat);
void flatmat_free_cuda(flatmat_t* mat);

#endif  // FLATMAT_H