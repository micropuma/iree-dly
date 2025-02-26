#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_MASK_WIDTH 3
#define TILE_SIZE 4


__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    // 考虑halo cell的情况
    /// 1. 计算当前线程的index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /// 2. 定义共享内存
    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
    int n = Mask_Width / 2;

    /// 3. 读取数据到共享内存
    // 读取左边的halo cell数据
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - n) {
        // 注意做ghost cell的边界处理
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left >= 0) ? N[halo_index_left] : 0;
    }

    // 读取右边的halo cell数据
    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) {
        // 注意做ghost cell的边界处理
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right < Width) ? N[halo_index_right] : 0;
    }
    // 读取中间的数据
    N_ds[threadIdx.x+n] = N[i];
    /// 4. 同步
    __syncthreads();

    /// 5. 计算卷积
    float Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++) {
        Pvalue += N_ds[threadIdx.x + j] * M[j];
    }

    P[i] = Pvalue;
}

int main() {
    const int Width = 16;   // Length of the input array N
    const int Mask_Width = 3;  // Length of the convolution mask M

    float N[Width] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};  // Input array
    float M[Mask_Width] = {0.25f, 0.5f, 0.25f};  // Convolution mask (e.g., a simple averaging mask)
    float P[Width];  // Output array

    // Device pointers
    float *d_N, *d_M, *d_P;

    // Allocate memory on the device
    cudaMalloc((void**)&d_N, Width * sizeof(float));
    cudaMalloc((void**)&d_M, Mask_Width * sizeof(float));
    cudaMalloc((void**)&d_P, Width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_N, N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = TILE_SIZE;  // You can adjust the block size based on your hardware
    int numBlocks = (Width + blockSize - 1) / blockSize;  // Calculate the number of blocks needed

    convolution_1D_basic_kernel<<<numBlocks, blockSize>>>(d_N, d_M, d_P, Mask_Width, Width);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the result from device to host
    cudaMemcpy(P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result of 1D Convolution: ";
    for (int i = 0; i < Width; i++) {
        std::cout << P[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}

