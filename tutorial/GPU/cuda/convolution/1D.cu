#include <iostream>
#include <cuda_runtime.h>

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0.0f;
    int N_start_point = i - (Mask_Width / 2);
    
    // Apply the mask
    for (int j = 0; j < Mask_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
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
    int blockSize = 256;  // You can adjust the block size based on your hardware
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

