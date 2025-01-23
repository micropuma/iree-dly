#!/bin/bash
iree-opt     --iree-stablehlo-to-iree-input \
             StableHLO2Input.mlir -o output.mlir \


