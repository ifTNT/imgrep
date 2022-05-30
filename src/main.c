#include <libbmp/libbmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct {
  int use_cpu;
  int block_size;
  int thread_size;
  char* src_path;
  char* tmpl_path;
} args;

void print_usage(const char* program_name) {
  fprintf(
      stderr,
      "Usage: %s [-c] [-b block_size] [-t thread_size] <src_img> <tmpl_img>\n"
      "-c : Use pure CPU to calculate. (block_size is ignored)\n"
      "-b block_size=1 : The block size of CUDA.\n"
      "-t thread_size=1 : (1.In CUDA mode) How many threads per block.\n"
      "                   (2.In CPU mode) How many threads will be launch.\n"
      "src_img : The path to the source image. (Only accept BMP format)\n"
      "tmpl_img : The path to the template image. (Only accept BMP format)\n",
      program_name);
}

int parse_args(int argc, char* argv[]) {
  int opt;

  args.use_cpu = 0;
  args.block_size = 1;
  args.thread_size = 1;

  while ((opt = getopt(argc, argv, "cb:t:")) != -1) {
    switch (opt) {
      case 'c':
        args.use_cpu = 1;
        break;
      case 'b':
        args.block_size = atoi(optarg);
        break;
      case 't':
        args.thread_size = atoi(optarg);
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

int main(int argc, char* argv[]) {
  if (parse_args(argc, argv) != 0) {
    exit(EXIT_FAILURE);
  }
  printf("use_cpu=%d; block_size=%d; thread_size=%d\n", args.use_cpu,
         args.block_size, args.thread_size);

  printf("Source image = %s\n", args.src_path);
  printf("Template image = %s\n", args.tmpl_path);

  /* Other code omitted */

  exit(EXIT_SUCCESS);
}