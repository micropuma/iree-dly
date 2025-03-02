#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16 // 2D Tiling block size

typedef struct {
    int width;
    int height;
    int pitch;  // This should store pitch in bytes, not elements
    int channels;
    float* data;
} ImageStruct, *wbImage_t;

__global__ void tiledMatrixKernel(float* input, float* output, int width, int height, int pitch) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // Padding to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * pitch / sizeof(float) + x]; // pitch is in bytes
    }
    __syncthreads();

    if (x < width && y < height) {
        output[y * pitch / sizeof(float) + x] = tile[threadIdx.y][threadIdx.x];
    }
}

void launchTiledKernel(wbImage_t img) {
    float *d_input, *d_output;
    size_t pitch_bytes;

    // Use cudaMallocPitch to ensure correct alignment
    cudaMallocPitch(&d_input, &pitch_bytes, img->width * sizeof(float), img->height);
    cudaMallocPitch(&d_output, &pitch_bytes, img->width * sizeof(float), img->height);
    
    // Copy data using cudaMemcpy2D
    cudaMemcpy2D(d_input, pitch_bytes, img->data, img->pitch * sizeof(float),
                 img->width * sizeof(float), img->height, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((img->width + TILE_SIZE - 1) / TILE_SIZE, (img->height + TILE_SIZE - 1) / TILE_SIZE);
    
    tiledMatrixKernel<<<gridDim, blockDim>>>(d_input, d_output, img->width, img->height, pitch_bytes);

    // Copy result back
    cudaMemcpy2D(img->data, img->pitch * sizeof(float), d_output, pitch_bytes,
                 img->width * sizeof(float), img->height, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int width = 32, height = 32;
    size_t pitch;
    
    wbImage_t img = new ImageStruct;
    img->width = width;
    img->height = height;
    img->channels = 1;
    
    // Allocate pitched memory for CPU
    img->data = (float*)malloc(height * width * sizeof(float));
    img->pitch = width; // Storing width as elements, not bytes
    
    for (int i = 0; i < height * width; ++i) {
        img->data[i] = static_cast<float>(i % 255);
    }

    launchTiledKernel(img);

    delete[] img->data;
    delete img;
    return 0;
}


