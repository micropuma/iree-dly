#include <stdio.h>
#include <cuda_runtime.h>

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device %d: %s\n", device, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA-compatible devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        printDeviceProperties(i);
    }
    
    return 0;
}
