#compilation0 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf16>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf16>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<?x?xf16>) -> tensor<?x?xf16>
  %result = linalg.matmul {compilation_info = #compilation0} ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf16>) -> tensor<?x?xf16>
  return %result: tensor<?x?xf16>
}

#compilation1 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_457x330xf16_times_330x512xf16_into_457x512xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<457x330xf16>, %rhs: tensor<330x512xf16>) -> tensor<457x512xf16> {
  %init_acc = tensor.empty() : tensor<457x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<457x512xf16>) -> tensor<457x512xf16>
  %result = linalg.matmul {compilation_info = #compilation1} ins(%lhs, %rhs: tensor<457x330xf16>, tensor<330x512xf16>) outs(%acc: tensor<457x512xf16>) -> tensor<457x512xf16>
  return %result: tensor<457x512xf16>
}

#compilation3 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_438x331xf16_times_331x513xf16_into_438x513xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<438x331xf16>, %rhs: tensor<331x513xf16>) -> tensor<438x513xf16> {
  %init_acc = tensor.empty() : tensor<438x513xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<438x513xf16>) -> tensor<438x513xf16>
  %result = linalg.matmul {compilation_info = #compilation3} ins(%lhs, %rhs: tensor<438x331xf16>, tensor<331x513xf16>) outs(%acc: tensor<438x513xf16>) -> tensor<438x513xf16>
  return %result: tensor<438x513xf16>
}

#compilation5 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_540x332xf16_times_332x516xf16_into_540x516xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<540x332xf16>, %rhs: tensor<332x516xf16>) -> tensor<540x516xf16> {
  %init_acc = tensor.empty() : tensor<540x516xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<540x516xf16>) -> tensor<540x516xf16>
  %result = linalg.matmul {compilation_info = #compilation5} ins(%lhs, %rhs: tensor<540x332xf16>, tensor<332x516xf16>) outs(%acc: tensor<540x516xf16>) -> tensor<540x516xf16>
  return %result: tensor<540x516xf16>
}

#compilation7 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_1000x4xf16_times_4x512xf16_into_1000x512xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<1000x4xf16>, %rhs: tensor<4x512xf16>) -> tensor<1000x512xf16> {
  %init_acc = tensor.empty() : tensor<1000x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<1000x512xf16>) -> tensor<1000x512xf16>
  %result = linalg.matmul {compilation_info = #compilation7} ins(%lhs, %rhs: tensor<1000x4xf16>, tensor<4x512xf16>) outs(%acc: tensor<1000x512xf16>) -> tensor<1000x512xf16>
  return %result: tensor<1000x512xf16>
}

#compilation9 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_4x1000xf16_times_1000x512xf16_into_4x512xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<4x1000xf16>, %rhs: tensor<1000x512xf16>) -> tensor<4x512xf16> {
  %init_acc = tensor.empty() : tensor<4x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<4x512xf16>) -> tensor<4x512xf16>
  %result = linalg.matmul {compilation_info = #compilation9} ins(%lhs, %rhs: tensor<4x1000xf16>, tensor<1000x512xf16>) outs(%acc: tensor<4x512xf16>) -> tensor<4x512xf16>
  return %result: tensor<4x512xf16>
}

#compilation11 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_512x1000xf16_times_1000x4xf16_into_512x4xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<512x1000xf16>, %rhs: tensor<1000x4xf16>) -> tensor<512x4xf16> {
  %init_acc = tensor.empty() : tensor<512x4xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<512x4xf16>) -> tensor<512x4xf16>
  %result = linalg.matmul {compilation_info = #compilation11} ins(%lhs, %rhs: tensor<512x1000xf16>, tensor<1000x4xf16>) outs(%acc: tensor<512x4xf16>) -> tensor<512x4xf16>
  return %result: tensor<512x4xf16>
}

#compilation13 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_513x128xf16_times_128x55xf16_into_513x55xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<513x128xf16>, %rhs: tensor<128x55xf16>) -> tensor<513x55xf16> {
  %init_acc = tensor.empty() : tensor<513x55xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<513x55xf16>) -> tensor<513x55xf16>
  %result = linalg.matmul {compilation_info = #compilation13} ins(%lhs, %rhs: tensor<513x128xf16>, tensor<128x55xf16>) outs(%acc: tensor<513x55xf16>) -> tensor<513x55xf16>
  return %result: tensor<513x55xf16>
}

#compilation15 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_7x160xf16_times_160x31xf16_into_7x31xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<7x160xf16>, %rhs: tensor<160x31xf16>) -> tensor<7x31xf16> {
  %init_acc = tensor.empty() : tensor<7x31xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<7x31xf16>) -> tensor<7x31xf16>
  %result = linalg.matmul {compilation_info = #compilation15} ins(%lhs, %rhs: tensor<7x160xf16>, tensor<160x31xf16>) outs(%acc: tensor<7x31xf16>) -> tensor<7x31xf16>
  return %result: tensor<7x31xf16>
}

#compilation17 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_512x330xf16_times_330x33xf16_into_512x33xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<512x330xf16>, %rhs: tensor<330x33xf16>) -> tensor<512x33xf16> {
  %init_acc = tensor.empty() : tensor<512x33xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<512x33xf16>) -> tensor<512x33xf16>
  %result = linalg.matmul {compilation_info = #compilation17} ins(%lhs, %rhs: tensor<512x330xf16>, tensor<330x33xf16>) outs(%acc: tensor<512x33xf16>) -> tensor<512x33xf16>
  return %result: tensor<512x33xf16>
}

#compilation18 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %acc: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %result = linalg.matmul {compilation_info = #compilation18} ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf16>) -> tensor<?x?xf16>

  return %result: tensor<?x?xf16>
}

#compilation19 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_accumulate_1x1000xf16_times_1000x1000xf16_into_1x1000xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<1x1000xf16>, %rhs: tensor<1000x1000xf16>, %acc: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
  %result = linalg.matmul {compilation_info = #compilation19} ins(%lhs, %rhs: tensor<1x1000xf16>, tensor<1000x1000xf16>) outs(%acc: tensor<1x1000xf16>) -> tensor<1x1000xf16>

  return %result: tensor<1x1000xf16>
}

#compilation21 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_accumulate_1000x1000xf16_times_1000x1xf16_into_1000x1xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<1000x1000xf16>, %rhs: tensor<1000x1xf16>, %acc: tensor<1000x1xf16>) -> tensor<1000x1xf16> {
  %result = linalg.matmul {compilation_info = #compilation21} ins(%lhs, %rhs: tensor<1000x1000xf16>, tensor<1000x1xf16>) outs(%acc: tensor<1000x1xf16>) -> tensor<1000x1xf16>

  return %result: tensor<1000x1xf16>
}

#compilation23 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3, store_stage = 1}>>func.func @matmul_1000x1000xf16_times_1000x1xf16_into_1000x1xf16_for_LLVMGPUMatmulTensorCore(%lhs: tensor<1000x1000xf16>, %rhs: tensor<1000x1xf16>) -> tensor<1000x1xf16> {
  %init_acc = tensor.empty() : tensor<1000x1xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<1000x1xf16>) -> tensor<1000x1xf16>
  %result = linalg.matmul {compilation_info = #compilation23} ins(%lhs, %rhs: tensor<1000x1000xf16>, tensor<1000x1xf16>) outs(%acc: tensor<1000x1xf16>) -> tensor<1000x1xf16>
  return %result: tensor<1000x1xf16>
}

