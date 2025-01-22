// %operand: [
//            [1, 2, 3],
//            [4, 5, 6]
//           ]
// %padding_value: 0
// %result = "stablehlo.pad"(%operand, %padding_value) {
//   edge_padding_low = array<i64: 0, 1>,
//   edge_padding_high = array<i64: 2, 1>,
//   interior_padding = array<i64: 1, 2>
// } : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// %result: [
//           [0, 1, 0, 0, 2, 0, 0, 3, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 4, 0, 0, 5, 0, 0, 6, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0]
//          ]

func.func @pad_test(%input: tensor<2x3xi32>, %c0: tensor<i32>) -> tensor<3x9xi32> {
  %res = "stablehlo.pad"(%input, %c0) {
    edge_padding_low = array<i64: 0, 1>,
    edge_padding_high = array<i64: 1, 5>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<2x3xi32>, tensor<i32>) -> tensor<3x9xi32>
  return %res : tensor<3x9xi32>
}

