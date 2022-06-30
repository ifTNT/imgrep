SRC_PATH = src
LIB_PATH = lib
CUDA_PATH = $(dir $(shell which nvcc))..

CFLAGS = -c -Wall -O2 -g
LDFLAGS = -L${CUDA_PATH}/lib64 -lpthread -lm -lcudadevrt -lcudart
CUDA_ARCH = -arch=sm_50
CUDA_CFLAGS := --device-c ${CUDA_ARCH} -G -forward-unknown-to-host-compiler ${CFLAGS}
CUDA_LDFLAGS := --device-link ${CUDA_ARCH} -lcudadevrt -lcudart
INCLUDE_PATH = -I${SRC_PATH} -I${LIB_PATH} -I${CUDA_PATH}/include
CC = gcc
NVCC = nvcc

HEADERS := $(wildcard ${SRC_PATH}/*.h)
CU_SRCS := $(wildcard ${SRC_PATH}/*.cu)
CU_OBJS := $(patsubst %.cu, %.cu.o, $(CU_SRCS))
LIBS    := $(dir $(wildcard ${LIB_PATH}/*/))
C_SRCS  := $(wildcard ${SRC_PATH}/*.c ${addsuffix *.c, ${LIBS}})
C_OBJS  := $(patsubst %.c, %.c.o, $(C_SRCS))
DEVICE_OBJ = ${SRC_PATH}/device_link.o
OBJS := ${C_OBJS} ${CU_OBJS} ${DEVICE_OBJ}
EXEC = imgrep

.PHONY: all clean
all: ${EXEC}

${EXEC}: ${OBJS}
	${CC} -o $@ $^ ${LDFLAGS}

# If want to use the host linker,
# it's necessary to do an intermediate device code link step.
${DEVICE_OBJ}: ${CU_OBJS}
	${NVCC} -o $@ $^ ${CUDA_LDFLAGS}

# Cancel implicit rules to prevent circular dependency
.SUFFIXES: 

%.cu.o: %.cu ${HEADERS}
	${NVCC} -o $@ ${CUDA_CFLAGS} ${INCLUDE_PATH} $<

%.c.o: %.c ${HEADERS}
	${CC} -o $@ ${CFLAGS} ${INCLUDE_PATH} $<

clean:
	rm -rf ${OBJS}
	rm -rf ${EXEC}
