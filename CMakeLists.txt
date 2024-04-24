cmake_minimum_required(VERSION 3.16)
project(top-stencil LANGUAGES C DESCRIPTION "3D stencil with MPI")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_C_STANDARD 17)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-Wall -g -O3 -march=native)
endif()

# MPI Setup
find_package(MPI REQUIRED)

# If MPI was found, include MPI compiler flags and include directories
if(MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
endif()

find_package(BLAS REQUIRED)


# Print the compiler being used
message("Compiler   : ${CMAKE_C_COMPILER}")
message("Build type : ${CMAKE_BUILD_TYPE}")

# Open MP setup
#find_package(OpenMP REQUIRED)

add_subdirectory(src)

add_executable(top-stencil src/main.c)

target_include_directories(top-stencil PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(top-stencil PRIVATE stencil::stencil stencil::utils)
