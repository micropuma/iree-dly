iree-opt --pass-pipeline="builtin.module(iree-dispatch-creation-fold-unit-extent-dims, iree-dispatch-creation-pipeline)" --split-input-file --mlir-print-local-scope pipeline.mlir -o temp3.mlir
