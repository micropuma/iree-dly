#!/bin/bash
iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --mlir-print-ir-after-all \
             example1.mlir -o example1.vmfb \
             2>&1 | tee output.dump

iree-run-module --device=cuda \
                --module=example1.vmfb \
                --function=add \
                --input="4xf32=[1 2 3 4]" \
                --input="4xf32=[2 2 2 2]"
