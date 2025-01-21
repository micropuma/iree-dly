func.func @pad_test(%input: tensor<2x3xi32>, %c0: tensor<i32>) -> tensor<3x9xi32> {
  %res = "stablehlo.pad"(%input, %c0) {
    edge_padding_low = array<i64: 0, 1>,
    edge_padding_high = array<i64: 1, 5>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<2x3xi32>, tensor<i32>) -> tensor<3x9xi32>
  return %res : tensor<3x9xi32>
}

