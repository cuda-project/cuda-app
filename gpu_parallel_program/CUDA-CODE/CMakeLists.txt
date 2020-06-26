cmake_minimum_required(VERSION 3.5)
project(cuda-app)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
file(GLOB CU imedgeG.cu)
cuda_add_executable(imedgeG imedgeG.cu)
