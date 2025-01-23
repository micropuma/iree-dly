func.func @main(%arg0: tensor<28x28xf32>, %arg1: tensor<784x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x10xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<28x28xf32>) -> tensor<784xf32>
  %1 = stablehlo.dot %0, %arg1 : (tensor<784xf32>, tensor<784x10xf32>) -> tensor<10xf32>
  %2 = stablehlo.reshape %1 : (tensor<10xf32>) -> tensor<1x10xf32>
  %3 = stablehlo.add %2, %arg2 : tensor<1x10xf32>
  %4 = stablehlo.maximum %3, %cst : tensor<1x10xf32>
  return %4 : tensor<1x10xf32>
}
