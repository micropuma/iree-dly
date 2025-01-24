#map = affine_map<()[s0] -> (s0 ceildiv 8)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @pack_elementwise_fusion(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = affine.apply #map()[%dim]
    %2 = affine.apply #map1()[%dim_0]
    %3 = tensor.empty(%1, %2) : tensor<?x?x8x32xf32>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %4 = flow.dispatch.region -> (tensor<?x?x8x32xf32>{%1, %2}) {
      %5 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : tensor<?x?xf32>, tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %6 = arith.addf %in, %in_3 : f32
        linalg.yield %6 : f32
      } -> tensor<?x?xf32>
      %pack = tensor.pack %5 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %3 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
      flow.return %pack : tensor<?x?x8x32xf32>
    }
    util.return %4 : tensor<?x?x8x32xf32>
  }
}

