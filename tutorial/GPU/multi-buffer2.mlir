// -----// IR Dump After GPUMultiBufferingPass (iree-codegen-gpu-multi-buffering) //----- //
func.func @matmul_accumulate_512x128xf16_times_128x512xf16_into_512x512xf16_for_LLVMGPUMatmulTensorCore_32_32_16_64_2_1_dispatch_0_matmul_512x512x128_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1], {pipeline_depth = 3 : i64, store_stage = 1 : i64}>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %alloc = memref.alloc() : memref<3x16x32xf16, #gpu.address_space<workgroup>>
  %alloc_0 = memref.alloc() : memref<3x32x16xf16, #gpu.address_space<workgroup>>
  %alloc_1 = memref.alloc() : memref<32x32xf16, #gpu.address_space<workgroup>>
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x128xf16, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %0, 64 : memref<512x128xf16, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<128x512xf16, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %1, 64 : memref<128x512xf16, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<512x512xf16, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %2, 64 : memref<512x512xf16, #hal.descriptor_type<storage_buffer>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %subview = memref.subview %0[%3, 0] [32, 128] [1, 1] : memref<512x128xf16, #hal.descriptor_type<storage_buffer>> to memref<32x128xf16, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %subview_2 = memref.subview %1[0, %4] [128, 32] [1, 1] : memref<128x512xf16, #hal.descriptor_type<storage_buffer>> to memref<128x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %subview_3 = memref.subview %2[%3, %4] [32, 32] [1, 1] : memref<512x512xf16, #hal.descriptor_type<storage_buffer>> to memref<32x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  gpu.barrier
  memref.copy %subview_3, %alloc_1 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<32x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<32x32xf16, #gpu.address_space<workgroup>>
  gpu.barrier
  scf.for %arg0 = %c0 to %c128 step %c16 {
    %5 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 3)>(%arg0)
    %subview_4 = memref.subview %alloc_0[%5, 0, 0] [1, 32, 16] [1, 1, 1] : memref<3x32x16xf16, #gpu.address_space<workgroup>> to memref<32x16xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
    %6 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 3)>(%arg0)
    %subview_5 = memref.subview %alloc[%6, 0, 0] [1, 16, 32] [1, 1, 1] : memref<3x16x32xf16, #gpu.address_space<workgroup>> to memref<16x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    %subview_6 = memref.subview %subview[0, %arg0] [32, 16] [1, 1] : memref<32x128xf16, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<32x16xf16, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
    %subview_7 = memref.subview %subview_2[%arg0, 0] [16, 32] [1, 1] : memref<128x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
    gpu.barrier
    memref.copy %subview_6, %subview_4 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<32x16xf16, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<32x16xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
    memref.copy %subview_7, %subview_5 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<16x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    gpu.barrier
    %thread_id_x = gpu.thread_id  x
    %thread_id_y = gpu.thread_id  y
    %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%thread_id_y]
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16)>(%thread_id_x)
    %subview_8 = memref.subview %subview_4[%7, 0] [16, 16] [1, 1] : memref<32x16xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
    %subview_9 = memref.subview %subview_5[0, %8] [16, 16] [1, 1] : memref<16x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>> to memref<16x16xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    %subview_10 = memref.subview %alloc_1[%7, %8] [16, 16] [1, 1] : memref<32x32xf16, #gpu.address_space<workgroup>> to memref<16x16xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>} ins(%subview_8, %subview_9 : memref<16x16xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<16x16xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%subview_10 : memref<16x16xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>)
  }
  gpu.barrier
  memref.copy %alloc_1, %subview_3 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<32x32xf16, #gpu.address_space<workgroup>> to memref<32x32xf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  gpu.barrier
  return
}
