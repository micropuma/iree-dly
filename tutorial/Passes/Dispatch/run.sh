iree-opt \
        --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" \
        --split-input-file \
        form-dispatch.mlir -o temp.mlir