cmake_minimum_required(VERSION 3.9)
project(p33)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

include_directories(
        src
)

set(SOURCE_FILES
        main.cu
        src/complex.h
        src/complex.cu
        src/input_image.h
        src/input_image.cc
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cu)
