module {
  func.func @foo(%arg0: !hal.buffer_view, %arg1: f32) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x1xf32>
    %1 = tensor.empty() : tensor<5x7xf32>
    %2 = linalg.fill ins(%arg1 : f32) outs(%1 : tensor<5x7xf32>) -> tensor<5x7xf32>
    %inserted_slice = tensor.insert_slice %0 into %2[1, 2] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<5x7xf32>
    %3 = hal.tensor.export %inserted_slice : tensor<5x7xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}

