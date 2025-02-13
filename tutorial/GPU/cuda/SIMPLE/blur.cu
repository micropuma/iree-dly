#include <stdio.h>
#include <cuda_runtime.h>

#define BLUR_SIZE 1  // 模糊核大小

// CUDA 模糊处理核函数
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        // 遍历 BLUR_SIZE x BLUR_SIZE 领域
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                // 确保当前像素有效
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }

        // 计算平均值并写入输出
        out[Row * w + Col] = (unsigned char)(pixVal / pixels);
    }
}

// 生成测试数据
void generateTestImage(unsigned char *image, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        image[i] = rand() % 256; // 随机生成 0-255 范围的灰度值
    }
}

// 打印图像（仅适用于小图）
void printImage(const unsigned char *image, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%3d ", image[i * w + j]);
        }
        printf("\n");
    }
}

int main() {
    int width = 8;
    int height = 8;
    int size = width * height * sizeof(unsigned char);

    // 分配主机内存
    unsigned char *h_in = (unsigned char *)malloc(size);
    unsigned char *h_out = (unsigned char *)malloc(size);

    // 生成测试图像
    generateTestImage(h_in, width, height);
    
    printf("Original Image:\n");
    printImage(h_in, width, height);

    // 分配设备内存
    unsigned char *d_in, *d_out;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // 复制数据到设备
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 设置 CUDA 线程块
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    // 执行 CUDA 核函数
    blurKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);

    // 复制结果回主机
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("\nBlurred Image:\n");
    printImage(h_out, width, height);

    // 释放内存
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

