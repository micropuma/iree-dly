iree-compile linalg.mlir \
             --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-hal-dump-executable-benchmarks-to=/benchmark/ \
             -o /dev/null