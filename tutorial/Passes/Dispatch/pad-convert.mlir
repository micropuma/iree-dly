// 测试IREE项目对于padding的处理流程
func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x1xf32>
  %padded = tensor.pad %0 low[1, 2] high[3, 4] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<1x1xf32> to tensor<5x7xf32>
  %1 = hal.tensor.export %padded : tensor<5x7xf32> -> !hal.buffer_view
  return %1 : !hal.buffer_view
}
