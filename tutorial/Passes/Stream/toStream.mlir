module {
  util.global private @_constant {noinline} = dense<[0.000000e+00, 0.00999999977, 2.000000e-02, 3.000000e-02, 4.000000e-02, 5.000000e-02, 6.000000e-02, 7.000000e-02, 8.000000e-02, 9.000000e-02]> : tensor<10xf32>
  flow.executable private @test_dispatch_0 {
    flow.executable.export public @test_dispatch_0_generic_10 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @test_dispatch_0_generic_10(%arg0: !flow.dispatch.tensor<readonly:tensor<10xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<10xf32>>, %arg2: !flow.dispatch.tensor<readwrite:tensor<10xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
        %2 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<10xf32>> -> tensor<10xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<10xf32>, tensor<10xf32>) outs(%2 : tensor<10xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %4 = arith.addf %in, %in_0 : f32
          linalg.yield %4 : f32
        } -> tensor<10xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<readwrite:tensor<10xf32>>
        return
      }
    }
  }
  func.func @test(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c10 = arith.constant 10 : index
    %_constant = util.global.load @_constant : tensor<10xf32>
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x10xf32>
    %1 = flow.tensor.reshape %0 : tensor<1x10xf32> -> tensor<10xf32>
    %2 = flow.tensor.empty : tensor<10xf32>
    %3 = flow.dispatch @test_dispatch_0::@test_dispatch_0_generic_10[%c10](%1, %_constant, %2) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> %2
    %4 = flow.tensor.reshape %3 : tensor<10xf32> -> tensor<1x10xf32>
    %5 = hal.tensor.export %4 : tensor<1x10xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}