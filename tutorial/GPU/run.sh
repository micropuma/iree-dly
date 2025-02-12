iree-compile --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  matmul.mlir \
  --mlir-disable-threading \
  --mlir-elide-elementsattrs-if-larger=10 \
  -o test.vmfb
