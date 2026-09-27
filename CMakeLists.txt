cmake_minimum_required(VERSION 3.18)

# Set architecture before enabling CUDA language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Default to Release if no build type was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(RayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(raytracer
    test.cu
    image.cu
)

# Apply optimizations specifically to the target for Release builds
target_compile_options(raytracer PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
target_include_directories(raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(raytracer PRIVATE CUDA::cudart)