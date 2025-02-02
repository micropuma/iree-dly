func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %c2 = arith.constant 2 : index
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2xf32>
  %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2xf32>
  %2 = flow.dispatch.workgroups[%c2](%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> =
        (%arg2: !flow.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: !flow.dispatch.tensor<readonly:tensor<2xf32>>, %arg4: !flow.dispatch.tensor<writeonly:tensor<2xf32>>) {
    %4 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
    %5 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
    %6 = tensor.empty() : tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<2xf32>, tensor<2xf32>) outs(%6 : tensor<2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<2xf32>
    flow.dispatch.tensor.store %7, %arg4, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
    flow.return
  } count(%arg2: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2
    flow.return %x, %y, %z : index, index, index
  }
  %3 = hal.tensor.export %2 : tensor<2xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}
