cmake_minimum_required(VERSION 3.25)
project(simd)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories($<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_executable(simd_base64 benchmarks/base64.cpp)