func.func @test(%arg0: f32, %arg1: f32) -> tensor<1xf32> {
  %0 = tensor.from_elements %arg0 : tensor<1xf32>
  %1 = tensor.from_elements %arg1 : tensor<1xf32>
  %2 = mhlo.add %0, %1 : tensor<1xf32>
  return %2 : tensor<1xf32>
}