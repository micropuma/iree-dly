#!/bin/bash
iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --mlir-print-ir-after-all \
             scalar-tensorfrom.mlir -o scalar-tensorfrom.vmfb \
             2>&1 | tee output1.dump

iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --mlir-print-ir-after-all \
             shape-conversion.mlir -o shape-conversion.vmfb \
             2>&1 | tee output2.dump