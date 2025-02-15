#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

constexpr int TILE_WIDTH = 16;

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
    __shared__ float d_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float d_N[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float Pvalue = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numTiles; ph++) {
        if (row < width && (ph * TILE_WIDTH + tx) < width)
            d_M[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        else
            d_M[ty][tx] = 0.0f;

        if ((ph * TILE_WIDTH + ty) < width && col < width)
            d_N[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
        else
            d_N[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += d_M[ty][k] * d_N[k][tx];
        }
        __syncthreads();
    }

    if (row < width && col < width) {
        K[row * width + col] = Pvalue;
    }
}

void MatrixMultiplyCUDA(float *h_M, float *h_N, float *h_K, float *h_K1, int width) {
    float *d_M, *d_N, *d_K;
    float *d_M1, *d_N1, *d_K1;
    int size = width * width * sizeof(float);

    cudaMalloc(&d_M, size); cudaMalloc(&d_N, size); cudaMalloc(&d_K, size);
    cudaMalloc(&d_M1, size); cudaMalloc(&d_N1, size); cudaMalloc(&d_K1, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M1, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N1, h_N, size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_K, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("MatrixMulKernel execution time: %f ms\n", time);

    cudaEventRecord(start);
    MatrixMulKernelTiled<<<gridDim, blockDim>>>(d_M1, d_N1, d_K1, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("MatrixMulKernelTiled execution time: %f ms\n", time);

    cudaMemcpy(h_K, d_K, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K1, d_K1, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M); cudaFree(d_N); cudaFree(d_K);
    cudaFree(d_M1); cudaFree(d_N1); cudaFree(d_K1);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

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
    int width = 256;
    float *h_M = (float *)malloc(width * width * sizeof(float));
    float *h_N = (float *)malloc(width * width * sizeof(float));
    float *h_K = (float *)malloc(width * width * sizeof(float));
    float *h_K1 = (float *)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; i++) {
        h_M[i] = rand() % 10;
        h_N[i] = rand() % 10;
    }

    MatrixMultiplyCUDA(h_M, h_N, h_K, h_K1, width);

    // printf("Matrix M:\n");
    // printMatrix(h_M, width);
    // printf("\nMatrix N:\n");
    // printMatrix(h_N, width);
    // printf("\nMatrix K (Result):\n");
    // printMatrix(h_K, width);

    if (isOk(h_K, h_K1, width)) {
        printf("Tiling is correct!\n");
    } else {
        printf("Tiling is wrong!\n");
    }

    free(h_M); free(h_N); free(h_K); free(h_K1);
    return 0;
}
