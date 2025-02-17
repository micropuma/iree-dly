#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

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

__global__ void MatrixMulKernelTensorCore(half *M, half *N, float *K, int width) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);

    int bx = blockIdx.x * 16;
    int by = blockIdx.y * 16;

    if (bx < width && by < width) {
        wmma::load_matrix_sync(a, M + by * width + bx, width);
        wmma::load_matrix_sync(b, N + bx * width + by, width);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(K + by * width + bx, c, width, wmma::mem_row_major);
    }
}

void MatrixMultiplyCUDA(float *h_M, float *h_N, float *h_K, float *h_K1, float *h_K2, int width) {
    float *d_M, *d_N, *d_K;
    float *d_K1, *d_K2;
    half *d_Mh, *d_Nh;
    int size = width * width * sizeof(float);
    int half_size = width * width * sizeof(half);

    cudaMalloc(&d_M, size); cudaMalloc(&d_N, size); cudaMalloc(&d_K, size);
    cudaMalloc(&d_K1, size); cudaMalloc(&d_K2, size);
    cudaMalloc(&d_Mh, half_size); cudaMalloc(&d_Nh, half_size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    half *h_Mh = (half *)malloc(half_size);
    half *h_Nh = (half *)malloc(half_size);

    for (int i = 0; i < width * width; i++) {
        h_Mh[i] = __float2half(h_M[i]);
        h_Nh[i] = __float2half(h_N[i]);
    }
    
    cudaMemcpy(d_Mh, h_Mh, half_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nh, h_Nh, half_size, cudaMemcpyHostToDevice);

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
    MatrixMulKernelTiled<<<gridDim, blockDim>>>(d_M, d_N, d_K1, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("MatrixMulKernelTiled execution time: %f ms\n", time);

    cudaEventRecord(start);
    MatrixMulKernelTensorCore<<<gridDim, blockDim>>>(d_Mh, d_Nh, d_K2, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("MatrixMulKernelTensorCore execution time: %f ms\n", time);

    cudaMemcpy(h_K, d_K, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K1, d_K1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K2, d_K2, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M); cudaFree(d_N); cudaFree(d_K);
    cudaFree(d_K1); cudaFree(d_K2); cudaFree(d_Mh); cudaFree(d_Nh);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_Mh); free(h_Nh);
}

int main() {
    int width = 256;
    float *h_M = (float *)malloc(width * width * sizeof(float));
    float *h_N = (float *)malloc(width * width * sizeof(float));
    float *h_K = (float *)malloc(width * width * sizeof(float));
    float *h_K1 = (float *)malloc(width * width * sizeof(float));
    float *h_K2 = (float *)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; i++) {
        h_M[i] = rand() % 10;
        h_N[i] = rand() % 10;
    }

    MatrixMultiplyCUDA(h_M, h_N, h_K, h_K1, h_K2, width);

    free(h_M); free(h_N); free(h_K); free(h_K1); free(h_K2);
    return 0;
}

