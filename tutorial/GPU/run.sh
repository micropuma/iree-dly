iree-compile --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  --mlir-disable-threading \
  --mlir-elide-elementsattrs-if-larger=10 \
  --mlir-print-ir-after-all \
  matmul.mlir -o test.vmfb \
  2>&1 | tee output.dump
