cmake_minimum_required(VERSION 3.10)

project(imgrep VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

FILE(GLOB LIBBMP_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libbmp/*.c
)

FILE(GLOB SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -Wall -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -O2 -march=native")
set(CMAKE_CUDA_FLAGS "-g -Wall -O0")
set(CMAKE_C_CREATE_STATIC_LIBRARY
  "<CMAKE_AR> cr <TARGET> <LINK_FLAGS> <OBJECTS>"
  "<CMAKE_RANLIB> <TARGET>"
  )

add_library(bmp ${LIBBMP_SRC})
set_target_properties(bmp PROPERTIES LANGUAGE C)
set_target_properties(bmp PROPERTIES LINKER_LANGUAGE C)
add_executable(${PROJECT_NAME} ${SRC})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME}
                      PROPERTIES CUDA_ARCHITECTURES "all"
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${PROJECT_NAME}
  PRIVATE Threads::Threads
  bmp
)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)