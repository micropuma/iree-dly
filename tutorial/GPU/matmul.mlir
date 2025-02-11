#compilation0 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>,
  translation_info = <LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  ,
  { pipeline_depth = 3,   store_stage = 1 }>>
func.func @matmul_accumulate_512x128xf32_times_128x512xf32_into_512x512xf32_for_LLVMGPUMatmulTensorCore_32_32_16_64_2_1(%lhs: tensor<512x128xf32>, %rhs: tensor<128x512xf32>, %acc: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %result = linalg.matmul {compilation_info = #compilation0} ins(%lhs, %rhs: tensor<512x128xf32>, tensor<128x512xf32>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>

  return %result: tensor<512x512xf32>
}
