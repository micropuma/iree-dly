#include <stdio.h>
#include <cuda_runtime.h>

// 定义 CUDA 错误检查宏
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA Error: %s in %s at line %d\n",      \
                    cudaGetErrorString(err), __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// CUDA 设备函数：向量加法
__global__ void add(const float* a, const float* b, float* c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 10;
    int size = N * sizeof(float);

    // 分配主机内存
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 启动 CUDA 内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // 同步确保计算完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 打印结果
    for(int i = 0; i < N; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
