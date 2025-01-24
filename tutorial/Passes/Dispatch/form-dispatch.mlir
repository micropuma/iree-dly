// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" --split-input-file %s | FileCheck %s

util.func public @pack_elementwise_fusion(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %4 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %5 = linalg.generic  {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %arg0 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 :f32) :
      %6 = arith.addf %b0, %b1 : f32
      linalg.yield %6 : f32
  } -> tensor<?x?xf32>
  %6 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%d0]
  %7 = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%d1]
  %8 = tensor.empty(%6, %7) : tensor<?x?x8x32xf32>
  // TODO(#12746) : The inner_tiles could be dynamic here. It is disabled
  // due to unrelated codegen issue.
  %9 = tensor.pack %5 padding_value(%cst : f32)
      inner_dims_pos = [0, 1] inner_tiles = [8, 32]
      into %8 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
  util.return %9 : tensor<?x?x8x32xf32>
}