# EMCSS-CUDA-Imgrep

CUDA template matching project of 2022 EMCSS

## Methods

- TBD

## Dependency

- CUDA
- CMake 3.23 or above

## Build

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Execute

```
./build/imgrep [-c] [-m matcher] [-b block_size] [-t thread_size] <src_img> <tmpl_img>
```

### Arguments

```
-c : Use pure CPU to calculate. (block_size is ignored)
-m matcher : Matcher to calculate similarity. (PCC/SSD)
-b block_size=1 : The block size of CUDA.
-t thread_size=1 : (In CUDA mode) How many threads per block.
                   (In CPU mode) How many threads will be launch.
src_img : The path to the source image. (Only accept BMP format)
tmpl_img : The path to the template image. (Only accept BMP format)
```

## Author

Yung-Hsiung Hu @2022
