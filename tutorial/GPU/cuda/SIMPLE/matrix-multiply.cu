#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// 矩阵乘法核函数
__global__ void MatrixMulKernel(float *M, float *N, float *K, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < width && col < width) {
        float Pvalue = 0;
        for (int k = 0; k < width; k++) {
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        K[row * width + col] = Pvalue;
    }        
}

void MatrixMultiplyCUDA(float *h_M, float *h_N, float *h_K, int width) {
    float *d_M, *d_N, *d_K;
    int size = width * width * sizeof(float);

    // 1. 设备端动态分配内存
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_K, size);

    // 2. 复制数据到 GPU
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // 3. 计算线程块和网格大小
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (width + blockDim.y - 1) / blockDim.y);

    // 4. 启动 CUDA 核函数
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_K, width);
    cudaDeviceSynchronize();

    // 5. 复制结果回 CPU
    cudaMemcpy(h_K, d_K, size, cudaMemcpyDeviceToHost);

    // 6. 释放 GPU 内存
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_K);
}

// 打印矩阵
void printMatrix(float *matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int width = 4;  // 矩阵大小 (width x width)
    
    // 1. 在 CPU 端分配并初始化矩阵
    float *h_M = (float *)malloc(width * width * sizeof(float));
    float *h_N = (float *)malloc(width * width * sizeof(float));
    float *h_K = (float *)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; i++) {
        h_M[i] = rand() % 10;  // 随机数 0~9
        h_N[i] = rand() % 10;
    }

    // 2. 调用 CUDA 进行矩阵乘法
    MatrixMultiplyCUDA(h_M, h_N, h_K, width);

    // 3. 打印结果
    printf("Matrix M:\n");
    printMatrix(h_M, width);

    printf("\nMatrix N:\n");
    printMatrix(h_N, width);

    printf("\nMatrix K (Result):\n");
    printMatrix(h_K, width);

    // 4. 释放 CPU 内存
    free(h_M);
    free(h_N);
    free(h_K);

    return 0;
}

