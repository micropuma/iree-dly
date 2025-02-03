func @conv(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>)
          -> tensor<1x112x112x32xf32> {
  %0 = mhlo.convolution(%input, %filter)
            dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
            window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]}
            {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
            : (tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}