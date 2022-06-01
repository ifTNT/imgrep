extern "C" {
#include <flatmat.h>
}

typedef unsigned int uint;
typedef unsigned char uchar;

void flatmat_init(flatmat_t *dst, unsigned int width, unsigned int height,
                  unsigned int layer) {
  uint size = width * height * layer * sizeof(float);
  dst->el = (float *)malloc(size);
  dst->width = width;
  dst->height = height;
  dst->layer = layer;
}

void flatmat_init_cuda(flatmat_t *dst, unsigned int width, unsigned int height,
                       unsigned int layer) {
  uint size = width * height * layer * sizeof(float);
  cudaMalloc((void **)&(dst->el), size);
  dst->width = width;
  dst->height = height;
  dst->layer = layer;
}

void flatmat_from_bmp(flatmat_t *dst, bmp_img src) {
  const int CHANNELS = 3;
  uint w = src.img_header.biWidth;
  uint h = src.img_header.biHeight;
  flatmat_init(dst, w, h, CHANNELS);

  for (uint y = 0; y < h; y++) {
    for (uint x = 0; x < w; x++) {
      for (uint ch = 0; ch < CHANNELS; ch++) {
        dst->el[y * w * CHANNELS + x * CHANNELS + ch] =
            ((float)((uchar *)(&src.img_pixels[y][x]))[ch]);
      }
    }
  }
  return;
}

void flatmat_free(flatmat_t *mat) {
  free(mat->el);
  mat->el = NULL;
}

void flatmat_free_cuda(flatmat_t *mat) {
  cudaFree(mat->el);
  mat->el = NULL;
}