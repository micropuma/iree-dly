iree-compile 
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_86\
  matmul.mlir \
  --mlir-disable-threading \
  --mlir-elide-elementsattrs-if-larger=10 \
  -o test.vmfb
