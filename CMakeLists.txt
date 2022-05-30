cmake_minimum_required(VERSION 3.10)

project(imgrep VERSION 1.0 LANGUAGES C CUDA)

FILE(GLOB LIBBMP_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libbmp/*.c
)

FILE(GLOB SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -Wall -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -O2 -march=native")
set(CMAKE_CUDA_FLAGS "-g -Wall -O0")

add_library(bmp ${LIBBMP_SRC})
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME}
  PRIVATE Threads::Threads
  bmp
)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)