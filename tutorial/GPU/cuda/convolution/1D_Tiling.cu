#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 4
#define MAX_MASK_WIDTH 3

// 旧版的卷积核实现
__global__ void convolution_1D_old_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];  // 共享内存数组

    int n = Mask_Width / 2;

    // 读取数据到共享内存
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - n) {
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left >= 0) ? N[halo_index_left] : 0;
    }

    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) {
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right < Width) ? N[halo_index_right] : 0;
    }

    // 读取当前块的数据到共享内存
    N_ds[threadIdx.x + n] = N[i];

    // 同步线程，确保所有数据都加载到共享内存
    __syncthreads();

    // 计算卷积
    float Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++) {
        Pvalue += N_ds[threadIdx.x + j] * M[j];
    }

    // 将结果写回输出
    P[i] = Pvalue;
}

// 新版的卷积核实现（使用constant memory）
__constant__ float constant_M[3];  // 卷积核

__global__ void convolution_1D_new_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE + 3 - 1];  // 共享内存数组的大小

    int n = Mask_Width / 2;

    // 读取数据到共享内存
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - n) {
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left >= 0) ? N[halo_index_left] : 0;
    }

    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) {
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right < Width) ? N[halo_index_right] : 0;
    }

    // 读取当前块的数据到共享内存
    N_ds[threadIdx.x + n] = N[i];

    // 同步线程，确保所有数据都加载到共享内存
    __syncthreads();

    // 计算卷积
    float Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++) {
        Pvalue += N_ds[threadIdx.x + j] * M[j];
    }

    // 将结果写回输出
    P[i] = Pvalue;
}

int main() {
    const int Width = 16;   // 输入数组长度
    const int Mask_Width = 3;  // 卷积核的长度

    float N[Width] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};  // 输入数组
    float M[Mask_Width] = {0.25f, 0.5f, 0.25f};  // 卷积核
    float P[Width];  // 输出数组

    // Device pointers
    float *d_N, *d_P;

    // Allocate memory on the device
    cudaMalloc((void**)&d_N, Width * sizeof(float));
    cudaMalloc((void**)&d_P, Width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_N, N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_M, M, Mask_Width * sizeof(float));  // 将卷积核传到constant memory

    // 设置计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. 测试旧版卷积核
    cudaEventRecord(start);  // 记录开始时间
    int blockSize = TILE_SIZE;
    int numBlocks = (Width + blockSize - 1) / blockSize;

    convolution_1D_old_kernel<<<numBlocks, blockSize>>>(d_N, constant_M, d_P, Mask_Width, Width);
    cudaDeviceSynchronize();  // 确保 kernel 完成
    cudaEventRecord(stop);  // 记录结束时间
    cudaEventSynchronize(stop);  // 等待计时器同步
    float timeOldKernel = 0;
    cudaEventElapsedTime(&timeOldKernel, start, stop);  // 计算耗时

    std::cout << "Time for old kernel: " << timeOldKernel << " ms" << std::endl;

    // 2. 测试新版卷积核
    cudaMemcpy(d_P, N, Width * sizeof(float), cudaMemcpyHostToDevice);  // 重新初始化输出数组

    cudaEventRecord(start);  // 记录开始时间
    convolution_1D_new_kernel<<<numBlocks, blockSize>>>(d_N, constant_M, d_P, Mask_Width, Width);
    cudaDeviceSynchronize();  // 确保 kernel 完成
    cudaEventRecord(stop);  // 记录结束时间
    cudaEventSynchronize(stop);  // 等待计时器同步
    float timeNewKernel = 0;
    cudaEventElapsedTime(&timeNewKernel, start, stop);  // 计算耗时

    std::cout << "Time for new kernel (with constant memory): " << timeNewKernel << " ms" << std::endl;

    // Compare results
    std::cout << "Time improvement: " << (timeOldKernel - timeNewKernel) << " ms" << std::endl;

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
    cudaFree(d_P);

    return 0;
}
