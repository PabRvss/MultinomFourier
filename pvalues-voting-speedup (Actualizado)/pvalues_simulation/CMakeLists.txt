cmake_minimum_required(VERSION 3.0)
project(pvalues-simulation C)

include_directories("/opt/homebrew/opt/opencl-headers/include/")
include_directories("/opt/homebrew/opt/opencl-hpp-headers/include/")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set(CMAKE_C_STANDARD 11)

add_executable(pvalues-simulation src/main.c)
