#!/bin/bash
iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --mlir-print-ir-after-all \
             conv-layer.mlir -o conv-layer.vmfb \
             2>&1 | tee conv-layer.dump
