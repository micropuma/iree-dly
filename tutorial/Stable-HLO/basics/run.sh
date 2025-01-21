#!/bin/bash
iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --mlir-print-ir-after-all \
             pad.mlir -o pad.vmfb \
             2>&1 | tee output.dump

iree-run-module --device=cuda \
                --module=pad.vmfb \
                --input=2x3xi32=[[1,2,3],[4,5,6]] \
                --input=i32=0

