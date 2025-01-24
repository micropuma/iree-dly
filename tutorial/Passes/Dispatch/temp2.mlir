#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @custom_op_producer_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %0 = flow.dispatch.region -> (tensor<?xf32>{%dim}) {
      %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.mulf %in, %in : f32
        linalg.yield %3 : f32
      } -> tensor<?x?xf32>
      %2 = iree_linalg_ext.custom_op{indexing_maps = [#map, #map1], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]} ins(%1 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>):
        %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x?xf32>) outs(%arg3 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.addf %in, %out : f32
          linalg.yield %4 : f32
        } -> tensor<?xf32>
        iree_linalg_ext.yield %3 : tensor<?xf32>
      } -> tensor<?xf32>
      flow.return %2 : tensor<?xf32>
    }
    util.return %0 : tensor<?xf32>
  }
}

