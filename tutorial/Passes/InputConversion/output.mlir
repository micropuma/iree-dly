#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<28x28xf32>, %arg1: tensor<784x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x10xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<28x28xf32> into tensor<784xf32>
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
    %2 = linalg.vecmat ins(%collapsed, %arg1 : tensor<784xf32>, tensor<784x10xf32>) outs(%1 : tensor<10xf32>) -> tensor<10xf32>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
    %3 = tensor.empty() : tensor<1x10xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %arg2 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%3 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.addf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1x10xf32>
    %5 = tensor.empty() : tensor<1x10xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_0 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%5 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.maximumf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1x10xf32>
    return %6 : tensor<1x10xf32>
  }
}

