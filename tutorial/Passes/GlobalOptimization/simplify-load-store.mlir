func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
util.global private mutable @param0 : tensor<1x2xf32>
util.global private @param1 : tensor<1x2xf32>
func.func @run(%arg0: !hal.buffer_view) attributes {iree.abi.stub} {
  %param0 = util.global.load @param0 : tensor<1x2xf32>
  %0 = hal.tensor.export %param0 : tensor<1x2xf32> -> !hal.buffer_view
  %1 = call @add(%0, %arg0) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %2 = hal.tensor.import %1 : !hal.buffer_view -> tensor<1x2xf32>
  util.global.store %2, @param0 : tensor<1x2xf32>
  %param0_0 = util.global.load @param0 : tensor<1x2xf32>
  %param1 = util.global.load @param1 : tensor<1x2xf32>
  %3 = hal.tensor.export %param0_0 : tensor<1x2xf32> -> !hal.buffer_view
  %4 = hal.tensor.export %param1 : tensor<1x2xf32> -> !hal.buffer_view
  %5 = call @add(%3, %4) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %6 = hal.tensor.import %5 : !hal.buffer_view -> tensor<1x2xf32>
  util.global.store %6, @param0 : tensor<1x2xf32>
  return
}
