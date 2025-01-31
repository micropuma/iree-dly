iree-opt \
        --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" \
        --split-input-file \
        producer-consumer.mlir -o temp1.mlir
