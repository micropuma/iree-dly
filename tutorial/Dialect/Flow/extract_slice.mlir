util.func public @extract_slice2(%arg0 : tensor<5x24x48xf32>) -> tensor<2x48xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x48xf32>
  util.return %0 : tensor<2x48xf32>
}

util.func public @extract_slice4(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<2x24xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, 2, 24] [1, %arg1, 1]
      : tensor<5x24x48xf32> to tensor<2x24xf32>
  util.return %0 : tensor<2x24xf32>
}
