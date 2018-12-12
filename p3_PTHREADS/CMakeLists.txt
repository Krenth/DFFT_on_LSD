cmake_minimum_required(VERSION 3.9)
project(p3PTHREAD)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

include_directories(
        src
)

set(SOURCE_FILES
        main.cc
        src/complex.h
        src/complex.cc
        src/input_image.h
        src/input_image.cc
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cc)
target_link_libraries(${PROJECT_NAME} Threads::Threads)
