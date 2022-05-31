CFLAGS = -c -Wall -O2 -g
LDFLAGS = -L/opt/cuda/lib64 -lpthread -lm -lcudadevrt -lcudart
CUDA_ARCH = sm_50
CUDA_CFLAGS := --device-c -arch=${CUDA_ARCH} -g -G
CUDA_LDFLAGS := --device-link -arch=${CUDA_ARCH} -lcudadevrt -lcudart
INCLUDE_PATH = -Isrc/ -Ilib/
CC = gcc
NVCC = nvcc

CU_SRCS := $(wildcard src/*.cu)
CU_OBJS := $(addsuffix .o, $(CU_SRCS))
C_SRCS := $(wildcard src/*.c lib/libbmp/*.c)
C_OBJS := $(addsuffix .o, $(C_SRCS))
DEVICE_OBJ = src/device_link.o
OBJS := ${C_OBJS} ${CU_OBJS} ${DEVICE_OBJ}
EXEC = imgrep

.PHONY: all clean
all: ${EXEC}

${EXEC}: ${OBJS}
	${CC} -o $@ $^ ${LDFLAGS}

${DEVICE_OBJ}: ${CU_OBJS}
	${NVCC} -o $@ $^ ${CUDA_LDFLAGS}

%.cu.o: %.cu
	${NVCC} -o $@ ${CUDA_CFLAGS} ${INCLUDE_PATH} $^

%.c.o: %.c
	${CC} -o $@ ${CFLAGS} ${INCLUDE_PATH} $^

clean:
	rm -rf ${OBJS}
	rm -rf ${EXEC}
