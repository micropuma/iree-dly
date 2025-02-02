#!/bin/bash

# 针对IREE::Util::createSimplifyGlobalAccessesPass
iree-opt --pass-pipeline='builtin.module(func.func(iree-util-simplify-global-accesses))' \
	simplify-load-store.mlir -o opt.mlir

# 针对IREE::Util::createFoldGlobalsPass
iree-opt --pass-pipeline='builtin.module(iree-util-fold-globals)' simplify-load-store2.mlir -o opt2.mlir
