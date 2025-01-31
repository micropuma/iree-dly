iree-opt \
        --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups))" \
        --split-input-file \
        form-dispatch-workgroups.mlir -o temp2.mlir
