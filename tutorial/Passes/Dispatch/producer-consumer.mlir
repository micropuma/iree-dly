util.func @custom_op_producer_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.mulf %b0, %b0 : f32
      linalg.yield %1 :f32
  } -> tensor<?x?xf32>
  %2 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]}
      ins(%0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?xf32>):
      %3 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%b0 : tensor<?x?xf32>) outs(%b1 : tensor<?xf32>) {
        ^bb1(%bb0 : f32, %bb1 : f32) :
          %4 = arith.addf %bb0, %bb1 : f32
          linalg.yield %4 : f32
      } -> tensor<?xf32>
      iree_linalg_ext.yield %3 : tensor<?xf32>
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}