#!/bin/bash
# 检查IREE::Flow::createTensorPadToTensorInsertSlicePass
iree-opt --iree-dispatch-creation-tensor-pad-to-tensor-insert-slice \
	 pad-convert.mlir 

iree-opt --iree-dispatch-creation-tensor-pad-to-tensor-insert-slice \
	 pad-failed.mlir -o pad-failed-out.mlir

# 检查mlir::createConvertElementwiseToLinalgPass

# 检查mlir::createConvertElementwiseToLinalgPass

# 检查mlir::createLinalgFoldUnitExtentDimsPass

# 检查createInterchangeGenericOpsPass

# 检查createFusionOfTensorOpsPass

# 检查createCaptureDispatchDynamicDimsPass
iree-opt --iree-flow-capture-dynamic-dims \
	tie-shape.mlir

# 检查IREE::Flow::createOutlineDispatchRegionsPass
iree-opt --iree-flow-outline-dispatch-regions \
	dispatch-outline.mlir 