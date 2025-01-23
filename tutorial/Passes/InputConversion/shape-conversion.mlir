func.func @test(%arg0: tensor<1x?xf32>, %arg1: tensor<?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = shape.dim %arg0, %c1 : tensor<1x?xf32>, index -> index
  %1 = shape.dim %arg1, %c0 : tensor<?xf32>, index -> index
  %2 = shape.add %0, %1 : index, index -> index
  return %2 : index
}