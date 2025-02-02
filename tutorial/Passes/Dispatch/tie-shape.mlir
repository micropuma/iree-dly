// func.func @test(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
//  %0 = mhlo.add %arg0, %arg1 : tensor<?xf32>
//  return %0 : tensor<?xf32>
// }
func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?xf32>{%0}
  %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
  %3 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<?xf32>{%2}
  %4 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>()[%c0, %0, %c1]
  %5 = flow.dispatch.workgroups[%4](%0, %1, %3, %0, %2, %0) : (index, tensor<?xf32>{%0}, tensor<?xf32>{%2}, index, index, index) -> tensor<?xf32>{%0} =
      (%arg2: index, %arg3: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: !flow.dispatch.tensor<writeonly:tensor<?xf32>>) {
    %7 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [%arg7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg7} -> tensor<?xf32>
    %8 = flow.dispatch.tensor.load %arg4, offsets = [0], sizes = [%arg6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg6} -> tensor<?xf32>
    %9 = tensor.empty(%arg7) : tensor<?xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7, %8 : tensor<?xf32>, tensor<?xf32>) outs(%9 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.addf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<?xf32>
    flow.dispatch.tensor.store %10, %arg8, offsets = [0], sizes = [%arg7], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg7}
    flow.return
  } count(%arg2: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2
    flow.return %x, %y, %z : index, index, index
  }
  %6 = hal.tensor.export %5 : tensor<?xf32>{%0} -> !hal.buffer_view
  return %6 : !hal.buffer_view
}
