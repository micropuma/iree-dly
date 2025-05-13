//========================================== LLVM GPU Tile And Distibute ========================================
// Step1
func.func @matmul_accumulate_512x128xf16_times_128x512xf16_into_512x512xf16_for_LLVMGPUMatmulTensorCore_32_32_16_64_2_1_dispatch_0_matmul_512x512x128_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1], {pipeline_depth = 3 : i64, store_stage = 1 : i64}>} {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<32x32xf16, #gpu.address_space<workgroup>>
  // 获取interface的binding，这是数组A 
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x128xf16, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %0, 64 : memref<512x128xf16, #hal.descriptor_type<storage_buffer>>
  // 这是数组B
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<128x512xf16, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %1, 64 : memref<128x512xf16, #hal.descriptor_type<storage_buffer>>
  // 这是数组C
  // 注意返回的是memref<512x512xf16, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<512x512xf16, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %2, 64 : memref<512x512xf16, #hal.descriptor_type<storage_buffer>>

  // 获取workgroup中的thread id，是二维的
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %subview = memref.subview %0[%3, 0] [32, 128] [1, 1] : memref<512x128xf16, #hal.descriptor_type<storage_buffer>> to memref<32x128xf16, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %subview_0 = memref.subview %1[0, %4] [128, 32] [1, 1] : memref<128x512xf16, #hal.descriptor_type<storage_buffer>> to memref<128x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  // C的subview，这个是external memory
  %subview_1 = memref.subview %2[%3, %4] [32, 32] [1, 1] : memref<512x512xf16, #hal.descriptor_type<storage_buffer>> to memref<32x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %subview_2 = memref.subview %alloc[0, 0] [%c32, %c32] [1, 1] : memref<32x32xf16, #gpu.address_space<workgroup>> to memref<?x?xf16, strided<[32, 1]>, #gpu.address_space<workgroup>>
  // 将C的subview拷贝到workgroup memory中，subview_2是workgroup memory
  memref.copy %subview_1, %subview_2 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<32x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<?x?xf16, strided<[32, 1]>, #gpu.address_space<workgroup>>
  // 计算matmul，核心计算逻辑
  linalg.matmul {__internal_linalg_transform__ = "workgroup_memory", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>} ins(%subview, %subview_0 : memref<32x128xf16, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<128x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%subview_2 : memref<?x?xf16, strided<[32, 1]>, #gpu.address_space<workgroup>>)
  // 将C的结果拷贝回到external memory中
  memref.copy %subview_2, %subview_1 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<?x?xf16, strided<[32, 1]>, #gpu.address_space<workgroup>> to memref<32x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return
}


