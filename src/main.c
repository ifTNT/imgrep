#include <libbmp/libbmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "benchmark.h"
#include "flatmat.h"
#include "match.h"
#include "match_pcc.h"
#include "match_ssd.h"

/**
 * Extern varibles of getopt()
 */
extern char *optarg;
extern int optind, opterr, optopt;

enum matcher_e { MATCHER_PCC = 0, MATCHER_SSD, MATCHER_CNT };
static const char *MATCHER_STRING[] = {"PCC", "SSD", ""};

static struct {
  int use_cpu;
  enum matcher_e matcher;
  int blk_size;
  int thrd_size;
  char *src_path;
  char *tmpl_path;
} args;

static void print_usage(const char *program_name) {
  fprintf(
      stderr,
      "Usage: %s [-c] [-m matcher] [-b block_size] [-t thread_size] <src_img> "
      "<tmpl_img>\n"
      "  -c : Use pure CPU to calculate. (block_size is ignored)\n"
      "  -m matcher : Matcher to calculate similarity. (PCC/SSD)\n"
      "  -b block_size=1 : The block size of CUDA.\n"
      "  -t thread_size=1 : (In CUDA mode) How many threads per block.\n"
      "                     (In CPU mode) How many threads will be launch.\n"
      "  src_img : The path to the source image. (Only accept BMP format)\n"
      "  tmpl_img : The path to the template image. (Only accept BMP format)\n",
      program_name);
}

static int parse_args(int argc, char *argv[]) {
  int opt;

  args.use_cpu = 0;
  args.matcher = MATCHER_PCC;
  args.blk_size = 1;
  args.thrd_size = 1;

  while ((opt = getopt(argc, argv, "cm:b:t:")) != -1) {
    switch (opt) {
      case 'c':
        args.use_cpu = 1;
        break;
      case 'm':
        if (strcmp(optarg, "PCC") == 0) {
          args.matcher = MATCHER_PCC;
        } else if (strcmp(optarg, "SSD") == 0) {
          args.matcher = MATCHER_SSD;
        };
        break;
      case 'b':
        args.blk_size = atoi(optarg);
        break;
      case 't':
        args.thrd_size = atoi(optarg);
        break;
      default: /* '?' */
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
  }

  if (optind + 1 >= argc) {
    fprintf(stderr, "Miss template image\n");
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  args.src_path = argv[optind];
  args.tmpl_path = argv[optind + 1];

  return EXIT_SUCCESS;
}

static int read_img(bmp_img *img, char *path) {
  int ret = bmp_img_read(img, path);
  if (ret != BMP_OK) {
    fprintf(stderr, "Filed to open bmp image %s (%d)\n", path, ret);
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (parse_args(argc, argv) != 0) {
    exit(EXIT_FAILURE);
  }
  printf("use_cpu=%d; matcher=%s; block_size=%d; thread_size=%d\n",
         args.use_cpu, MATCHER_STRING[args.matcher], args.blk_size,
         args.thrd_size);

  // Initialize the matchers
  matcher_iface matchers[MATCHER_CNT];
  pcc_init(&matchers[MATCHER_PCC]);
  ssd_init(&matchers[MATCHER_SSD]);

  // Read the bmp file
  bmp_img src_bmp;
  bmp_img tmpl_bmp;
  flatmat_t src_img;
  flatmat_t tmpl_img;
  read_img(&src_bmp, args.src_path);
  read_img(&tmpl_bmp, args.tmpl_path);
  flatmat_from_bmp(&src_img, src_bmp);
  flatmat_from_bmp(&tmpl_img, tmpl_bmp);
  bmp_img_free(&src_bmp);
  bmp_img_free(&tmpl_bmp);

  printf("Source image: %s (%d, %d)\n", args.src_path, src_img.height,
         src_img.width);
  printf("Template image: %s (%d, %d)\n", args.tmpl_path, tmpl_img.height,
         tmpl_img.width);

  // Perform the benchmark of selected matcher
  double elapsed;
  match_result_t result;
  match_pos_t *pos;
  if (args.use_cpu) {
    benchmark_begin();
    matchers[args.matcher].match_cpu(&result, &src_img, &tmpl_img,
                                     args.thrd_size);
    elapsed = benchmark_end();
  } else {
    benchmark_begin();
    matchers[args.matcher].match_gpu(&result, &src_img, &tmpl_img,
                                     args.blk_size, args.thrd_size);
    elapsed = benchmark_end();
  }

  match_get_pos_list(&pos, &result);

  // Print out the result
  printf("Founded traget (cnt=%d):\n", result.cnt);
  for (int i = 0; i < result.cnt; i++) {
    printf("(%d, %d) ", pos[i].y, pos[i].x);
  }
  printf("\nFinished. Elapsed: %f seconds\n", elapsed);

  // Release resources
  match_free_pos_list(pos);
  match_free_result(&result);
  flatmat_free(&src_img);
  flatmat_free(&tmpl_img);

  exit(EXIT_SUCCESS);
}