C_FLAGS = -g -Wall -O0
INCLUDE_PATH = -Isrc/ -Ilib/
LD_FLAGS =  -L/opt/cuda/lib64 -lpthread -lm -lcuda -lcudart
CUDA_FLAGS = -dc -x cu -rdc=true
CC = g++
NVCC = nvcc

CU_SRCS := $(wildcard src/*.cu)
CU_OBJS := $(addsuffix .o, $(CU_SRCS))
C_SRCS := $(wildcard src/*.c lib/libbmp/*.c)
C_OBJS := $(addsuffix .o, $(C_SRCS))
DEVICE_LINK_OBJ = src/device_link.o
EXEC = imgrep

.PHONY: all clean
all: ${EXEC}

${EXEC}: ${DEVICE_LINK_OBJ} ${C_OBJS}
	${CC} ${LD_FLAGS} -o $@ $^

${DEVICE_LINK_OBJ}: ${CU_OBJS}
	${NVCC} -dlink ${LD_FLAGS} -o $@ $^

%.cu.o: %.cu
	${NVCC} -c ${CUDA_FLAGS} ${INCLUDE_PATH} -o $@ $^

%.c.o: %.c
	${CC} -c ${C_FLAGS} ${INCLUDE_PATH} -o $@ $^

clean:
	rm -rf ${CU_OBJS} ${C_OBJS} ${DEVICE_LINK_OBJ}
	rm -rf ${EXEC}
