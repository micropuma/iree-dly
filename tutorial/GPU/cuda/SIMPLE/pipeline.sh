nvcc shared-memory.cu -v -o shared-memory-arch_sm_80 --gpu-architecture=sm_80 、
    2>&1 | tee output.txt