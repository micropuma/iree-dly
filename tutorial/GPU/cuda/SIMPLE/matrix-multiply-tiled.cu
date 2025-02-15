#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

constexpr int TILE_WIDTH = 2;

// 矩阵乘法核函数（普通实现）
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

__global__ void MatrixMulKernelTiled(float *M, float *N, float *K, int width) {
    // 定义共享内存
    __shared__ float d_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float d_N[TILE_WIDTH][TILE_WIDTH];

    // 获取线程和块索引
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // 计算当前线程负责的全局行索引和列索引
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // 初始化 Pvalue
    float Pvalue = 0.0f;

    // 计算要迭代的 tile 数量
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;  // ceil(width / TILE_WIDTH)

    for (int ph = 0; ph < numTiles; ph++) {
        // 边界检查，防止越界访问
        if (row < width && (ph * TILE_WIDTH + tx) < width)
            d_M[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        else
            d_M[ty][tx] = 0.0f;  // 超出边界，填充 0

        if ((ph * TILE_WIDTH + ty) < width && col < width)
            d_N[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
        else
            d_N[ty][tx] = 0.0f;  // 超出边界，填充 0

        // 线程同步，确保所有线程都加载了数据
        __syncthreads();

        // 计算部分乘积
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += d_M[ty][k] * d_N[k][tx];
        }

        // 线程同步，确保当前 tile 计算完成后再加载下一组数据
        __syncthreads();
    }

    // 写回全局内存时也需要检查边界，防止写入非法地址
    if (row < width && col < width) {
        K[row * width + col] = Pvalue;
    }
}


void MatrixMultiplyCUDA(float *h_M, float *h_N, float *h_K, float *h_K1, int width) {
    float *d_M, *d_N, *d_K;
    float *d_M1, *d_N1, *d_K1;
    int size = width * width * sizeof(float);

    // 1. 设备端动态分配内存
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_K, size);
    
    cudaMalloc(&d_M1, size);
    cudaMalloc(&d_N1, size);
    cudaMalloc(&d_K1, size);

    // 2. 复制数据到 GPU
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_M1, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N1, h_N, size, cudaMemcpyHostToDevice);

    // 3. 计算线程块和网格大小
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    // 4. 启动 CUDA 核函数
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_K, width);
    cudaDeviceSynchronize();

    MatrixMulKernelTiled<<<gridDim, blockDim>>>(d_M1, d_N1, d_K1, width);
    cudaDeviceSynchronize();

    // 5. 复制结果回 CPU
    cudaMemcpy(h_K, d_K, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K1, d_K1, size, cudaMemcpyDeviceToHost);

    // 6. 释放 GPU 内存
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_K);
    cudaFree(d_M1);
    cudaFree(d_N1);
    cudaFree(d_K1);
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

bool isOk(float *matrix, float *matrix1, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (abs(matrix[i * width + j] - matrix1[i * width + j]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int width = 4;  // 矩阵大小 (width x width)
    
    // 1. 在 CPU 端分配并初始化矩阵
    float *h_M = (float *)malloc(width * width * sizeof(float));
    float *h_N = (float *)malloc(width * width * sizeof(float));
    float *h_K = (float *)malloc(width * width * sizeof(float));
    float *h_K1 = (float *)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; i++) {
        h_M[i] = rand() % 10;
        h_N[i] = rand() % 10;
    }

    // 2. 调用 CUDA 进行矩阵乘法
    MatrixMultiplyCUDA(h_M, h_N, h_K, h_K1, width);

    // 3. 打印结果
    printf("Matrix M:\n");
    printMatrix(h_M, width);

    printf("\nMatrix N:\n");
    printMatrix(h_N, width);

    printf("\nMatrix K (Result):\n");
    printMatrix(h_K, width);

    if (isOk(h_K, h_K1, width)) {
        printf("Tiling is correct!\n");
    } else {
        printf("Tiling is wrong!\n");
    }

    // 4. 释放 CPU 内存
    free(h_M);
    free(h_N);
    free(h_K);
    free(h_K1);

    return 0;
}
