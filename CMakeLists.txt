cmake_minimum_required(VERSION 3.14)
project(OpenCLBench VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OPENCLBENCH_BUILD_TESTS "Build stress and unit tests" OFF)
option(OPENCLBENCH_FETCH_OPENCL "Fetch Khronos OpenCL headers and C++ bindings" ON)

if(OPENCLBENCH_FETCH_OPENCL)
    include(FetchContent)
    
    FetchContent_Declare(
        OpenCL-Headers
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
        GIT_TAG        main
    )
    FetchContent_MakeAvailable(OpenCL-Headers)

    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)

    FetchContent_Declare(
        OpenCL-CLHPP
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
        GIT_TAG        main
    )
    FetchContent_MakeAvailable(OpenCL-CLHPP)
    
    set(OPENCL_INCLUDE_DIRS
        ${opencl-headers_SOURCE_DIR}
        ${opencl-clhpp_SOURCE_DIR}/include
    )
else()
    find_package(OpenCL REQUIRED)
    set(OPENCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
endif()

find_package(OpenCL)
if(NOT OpenCL_FOUND)
    message(WARNING "System OpenCL library not found. Linking might fail at runtime if not provided.")
endif()

set(SOURCES
    src/main.cpp
    src/core/device_manager.cpp
    src/core/tuner.cpp
    src/core/benchmark_runner.cpp
    src/core/reporting.cpp
    src/cli/cli_parser.cpp
    
    src/benchmarks/compute/black_scholes.cpp
    src/benchmarks/compute/mandelbrot.cpp
    src/benchmarks/compute/binomial_options.cpp
    src/benchmarks/compute/sobol_qrng.cpp
    src/benchmarks/compute/quasirandom_generator.cpp
    
    src/benchmarks/memory/bandwidth.cpp
    src/benchmarks/memory/fdtd3d.cpp
    src/benchmarks/memory/bicubic_texture.cpp
    src/benchmarks/memory/volume_filtering.cpp
    
    src/benchmarks/simulation/nbody.cpp
    src/benchmarks/simulation/smoke_particles.cpp
    src/benchmarks/simulation/fluids.cpp
    
    src/benchmarks/dsp/fwt.cpp
    src/benchmarks/dsp/convolution_fft2d.cpp
    src/benchmarks/dsp/haar_wavelet_1d.cpp
    
    src/benchmarks/image/sobel.cpp
    src/benchmarks/image/bilateral_filter.cpp
    src/benchmarks/image/recursive_gaussian.cpp
    src/benchmarks/image/nv12_to_bgr.cpp
    src/benchmarks/image/stereo_disparity.cpp
    
    src/benchmarks/interop/opengl_interop.cpp
    src/benchmarks/interop/vulkan_interop.cpp
)

add_executable(OpenCLBench ${SOURCES})

target_include_directories(OpenCLBench PRIVATE 
    src
    ${OPENCL_INCLUDE_DIRS}
)

if(OpenCL_FOUND)
    target_link_libraries(OpenCLBench PRIVATE OpenCL::OpenCL)
else()
    if(UNIX AND NOT APPLE)
        target_link_libraries(OpenCLBench PRIVATE OpenCL)
    endif()
endif()

file(GLOB_RECURSE KERNEL_FILES "${CMAKE_SOURCE_DIR}/kernels/*.cl")
foreach(KERNEL_FILE ${KERNEL_FILES})
    file(RELATIVE_PATH KERNEL_REL_PATH "${CMAKE_SOURCE_DIR}/kernels" "${KERNEL_FILE}")
    get_filename_component(KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels/${KERNEL_REL_PATH}" DIRECTORY)
    file(MAKE_DIRECTORY ${KERNEL_DIR})
    add_custom_command(
        TARGET OpenCLBench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${KERNEL_FILE}"
        "${CMAKE_BINARY_DIR}/kernels/${KERNEL_REL_PATH}"
    )
endforeach()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(OpenCLBench PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
elseif(MSVC)
    target_compile_options(OpenCLBench PRIVATE /O2 /W4 /MD)
endif()
