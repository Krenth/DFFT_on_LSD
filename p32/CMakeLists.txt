cmake_minimum_required(VERSION 3.9)
project(p30)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

include_directories(
        src
        ${MPI_INCLUDE_PATH}
)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(SOURCE_FILES
        main.cc
        src/complex.h
        src/complex.cc
        src/input_image.h
        src/input_image.cc
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cc)
