# OpenCLBench

OpenCLBench is a pfully native, cross-platform GPU benchmark written in modern C++17 and OpenCL. It is designed to evaluate OpenCL device performance across a variety of domains—including raw compute throughput, memory bandwidth, large-scale particle simulation, digital signal processing (DSP), and hardware-accelerated image processing.

Unlike standard graphics-focused benchmarks, OpenCLBench relies entirely on compute kernels. It is heavily modular, mathematically rigorous, and automatically adapts its workload parameters to optimize execution across different GPU architectures.

<img width="958" height="516" alt="demo" src="https://github.com/user-attachments/assets/ccfa3379-3d93-48c8-b179-57f42f76e08b" />

## Key Features

- **100% Native OpenCL**: Bypasses vendor-proprietary APIs (CUDA, HIP, Metal) to ensure unbiased, standardized compute evaluation.
- **Cross-Platform & Vendor Agnostic**: Works on Linux and Windows. It automatically detects and natively targets NVIDIA, AMD, Intel, and any other OpenCL 2.0+ compliant hardware.
- **Adaptive Optimization Engine**: Dynamically tunes local workgroup parameters (`local_work_size`, vector widths, and unroll factors) based on the specific hardware architecture it runs on (e.g., detecting wavefront sizes for AMD vs. warp sizes for NVIDIA).
- **Precise Profiling metrics**: Relies on OpenCL device-side hardware events and high-precision host timers to provide sub-millisecond metrics, factoring out kernel compile times and PCI-e transfer overhead.

## The Benchmarks

OpenCLBench is categorized into domain-specific benchmarks. Each benchmark isolates a specific subsystem of the GPU.

### 1. Compute: BlackScholes, Mandelbrot, BinomialOptions, SobolQRNG, QuasirandomGenerator
- **BlackScholes & BinomialOptions**: Parallel mathematical workloads simulating European options pricing.
- **Mandelbrot**: Computes the Mandelbrot set up to 256 iterations.
- **SobolQRNG & QuasirandomGenerator**: Massive pseudo-random number generator workloads via bitwise math.
- **Metrics output**: GFLOPS (Giga-Floating Point Operations Per Second).

### 2. Memory: Global Bandwidth, FDTD3d, BicubicTexture, VolumeFiltering
- **Global Bandwidth**: A memory-bound test using `float4` vector types.
- **FDTD3d**: A Finite-Difference Time-Domain 3D stencil.
- **BicubicTexture & VolumeFiltering**: Employs OpenCL Image samplers for highly spatial reads across 2D and 3D memory.
- **Metrics output**: GB/s (Gigabytes Per Second) and GFLOPS.

### 3. Simulation: NBody Gravity, SmokeParticles, Fluids
- **NBody Gravity & Fluids (SPH)**: Heavy $O(N^2)$ algorithms simulating particle interaction.
- **SmokeParticles**: Massive parallel advection grid updates.
- **Measures**: Mixed compute-memory workloads, `__local` memory caching efficiency.
- **Metrics output**: GFLOPS.

### 4. DSP: Fast Walsh Transform (FWT), ConvolutionFFT2D, HaarWavelet1D
- **Algorithms**: Standard frequency, image transform, and signal decomposition workloads testing global memory stride access and butterfly branching.
- **Metrics output**: GFLOPS and GB/s.

### 5. Image Processing: Sobel, Bilateral, RecursiveGaussian, NV12toBGR, StereoDisparity
- **Workloads**: Applies various heavy edge-detection, blurring, spatial-spatial filters, and format conversion matrices. 
- **Measures**: OpenCL `image2d_t` object efficiency, texture sampler performance.
- **Metrics output**: GFLOPS and GB/s.

### 6. Interop: OpenGL and Vulkan capabilities
- **Measures**: Automatically detects the presence of `cl_khr_gl_sharing` and `cl_khr_external_memory` for cross-API synchronization.


## Build Instructions

### Prerequisites
- CMake 3.14+
- A C++17 compliant compiler (GCC 7+ or Clang 5+)
- System OpenCL loader

### Installation (Ubuntu/Debian)
Run the following to install all necessary build tools, OpenCL loaders, Intel drivers, and diagnostic tools:
```bash
sudo apt update
sudo apt install -y build-essential cmake ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo intel-opencl-icd graphviz
```
*(Note: `intel-opencl-icd` is required for Intel GPUs. NVIDIA and AMD users should ensure their proprietary drivers are installed via the standard 'Software & Updates' tool or vendor installers).*

### Compiling
```bash
git clone https://github.com/Igriscodes/OpenCLBench.git
cd OpenCLBench

# Create a build directory
mkdir build && cd build

# Configure CMake (Downloads Khronos headers automatically)
cmake ..

# Build using all available cores
cmake --build . -j$(nproc)
```

*(Note: The build process automatically copies the `.cl` kernel files into the output directory so the application can JIT-compile them natively at runtime).*

## Usage Guide

OpenCLBench uses a clean, intuitive command-line interface.

### Command-Line Options

| Short | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-d` | `--device <index>` | Select a specific OpenCL device by index. | `0` |
| `-a` | `--all-devices` | Run the specified benchmarks sequentially across all available devices. | |
| `-l` | `--list` | List all available OpenCL platforms and devices on the system. | |
| `-c` | `--category <name>` | Run a specific category (`compute`, `memory`, `simulation`, `dsp`, `image`, `interop`, `all`). | `all` |
| `-b` | `--benchmark <name>` | Run a specific benchmark by its exact class name (e.g., `NBody`). | |
| `-i` | `--iterations <num>` | The number of iterations to run each kernel (ignored in stress mode). | `10` |
| | `--stress` | Enables Stress/Thermal Stability mode (runs continuous loops for 30s to measure thermal degradation). | |
| | `--json <path>` | Export the benchmark report to a JSON file. | |
| `-h` | `--help` | Print the help message. | |

### Examples

**Discover Devices**
```bash
./OpenCLBench -l
```

**Run All Benchmarks (Default)**
```bash
./OpenCLBench
```

**Run on All Devices**
```bash
./OpenCLBench -a
```

**Select a Specific Device**
```bash
./OpenCLBench -d 1
```

**Run a Specific Category or Benchmark**
```bash
# Run only memory benchmarks
./OpenCLBench -c memory

# Run specifically the NBody simulation
./OpenCLBench -b NBody
```

**Stress & Thermal Stability Mode**
Stress mode loops kernels continuously for a set duration (30 seconds per benchmark) and records thermal degradation (performance drop from the start vs end of the loop). You can combine this with category or benchmark selection.
```bash
# Stress test only the compute category
./OpenCLBench --stress -c compute

# Run the full suite in stress mode
./OpenCLBench --stress
```

**Export Results to JSON**
```bash
./OpenCLBench --json benchmark_report.json
```

## Project Architecture

- `src/main.cpp`: Entry point orchestrating the CLI and execution pipeline.
- `src/core/`: 
  - `device_manager`: Enumerates and stores capabilities of host GPUs.
  - `tuner`: Modifies dispatch dimensions and compiler pragmas based on device topology.
  - `benchmark_runner`: Aggregates timings, handles warm-up passes, and calculates standard deviations.
  - `reporting`: Formats the tabular terminal UI and machine-readable exports.
- `src/benchmarks/`: Host-side C++ objects responsible for buffer allocations, enqueueing, and defining metric formulas for each test.
- `kernels/`: Raw OpenCL C99/C11 `.cl` files containing the math and logic executed on the GPU.
