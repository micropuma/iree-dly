util.func public @tensor_reshape(%arg0 : tensor<?x4x?x5x?x6xf32>, %arg1 : tensor<20x?x40xf32>)
    -> (tensor<?x5x?xf32>, tensor<5x4x?x4x2x4x5xf32>)
{
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2], [3], [4, 5]]
      : tensor<?x4x?x5x?x6xf32> into tensor<?x5x?xf32>
  %1 = tensor.expand_shape %arg1 [[0, 1], [2, 3], [4, 5, 6]] output_shape [5, 4, 5, 4, 2, 4, 5]
      : tensor<20x?x40xf32> into tensor<5x4x?x4x2x4x5xf32>
  util.return %0, %1 : tensor<?x5x?xf32>, tensor<5x4x?x4x2x4x5xf32>
}
