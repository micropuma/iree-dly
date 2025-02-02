func.func @test(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %c40 = arith.constant 40 : index
  %c10 = arith.constant 10 : index
  %c553648160_i32 = arith.constant 553648160 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %_constant = util.global.load @_constant : !stream.resource<constant>
  hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("tensor") shape([%c1, %c10]) type(%c553648160_i32) encoding(%c1_i32)
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<1x10xf32> in !stream.resource<external>{%c40}
  %1 = stream.async.alloca : !stream.resource<external>{%c40}
  %2 = stream.async.dispatch @test_dispatch_0::@test_dispatch_0_generic_10[%c10](%0[%c0 to %c40 for %c40], %_constant[%c0 to %c40 for %c40], %1[%c0 to %c40 for %c40]) : (!stream.resource<external>{%c40}, !stream.resource<constant>{%c40}, !stream.resource<external>{%c40}) -> %1{%c40}
  %3 = stream.tensor.export %2 : tensor<1x10xf32> in !stream.resource<external>{%c40} -> !hal.buffer_view
  return %3 : !hal.buffer_view
}
