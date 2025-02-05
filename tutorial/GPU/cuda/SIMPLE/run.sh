nvcc shared-memory.cu -o shared-memory-base && 
nvcc shared-memory.cu -o shared-memory-arch_compute_70 --gpu-architecture=compute_70 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80 --gpu-architecture=compute_80 &&
nvcc shared-memory.cu -o shared-memory-arch_sm_80 --gpu-architecture=sm_80 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_compute_80 --gpu-architecture=compute_80 --gpu-code=compute_80 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_compute_80_sm_80 --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_compute_80_sm_80_sm_86 --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80,sm_86 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_compute_80_sm_86 --gpu-architecture=compute_80 --gpu-code=compute_80,sm_86 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_sm_80 --gpu-architecture=compute_80 --gpu-code=sm_80 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_sm_80_sm_86 --gpu-architecture=compute_80 --gpu-code=sm_80,sm_86 &&
nvcc shared-memory.cu -o shared-memory-arch_compute_80-code_sm_86 --gpu-architecture=compute_80 --gpu-code=sm_86