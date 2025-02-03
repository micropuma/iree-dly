# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      304   | 1.3%                             |
| 2: GMEM Coalescing                  |     2069.1 | 8.8%                             |
| 3: SMEM Caching                     |     2910.3 | 12.3%                            |
| 4: 1D Blocktiling                   |     8367.3 | 35.4%                            |
| 5: 2D Blocktiling                   |    15858.6 | 67.2%                            |
| 7: Avoid Bank Conflicts (Linearize) |    15979.7 | 67.7%                            |
| 8: Avoid Bank Conflicts (Offset)    |    16188.8 | 68.6%                            |
| 6: Vectorized Mem Access            |    17934.5 | 76.0%                            |
| 9: Autotuning                       |    19511.9 | 82.6%                            |
| 10: Warptiling                      |    21472.6 | 90.9%                            |
| 0: cuBLAS                           |    23612.1 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
