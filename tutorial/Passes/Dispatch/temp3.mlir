module {
  util.func public @main(%arg0: tensor<833xi32>, %arg1: tensor<833x833xf32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = flow.dispatch.workgroups(%arg0, %arg1, %arg2) : (tensor<833xi32>, tensor<833x833xf32>, tensor<f32>) -> tensor<f32> =
        (%arg3: !flow.dispatch.tensor<readonly:tensor<833xi32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<833x833xf32>>, %arg5: !flow.dispatch.tensor<readonly:tensor<f32>>, %arg6: !flow.dispatch.tensor<writeonly:tensor<f32>>) {
      %cst = arith.constant 5.66893432E-4 : f32
      %1 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [833], strides = [1] : !flow.dispatch.tensor<readonly:tensor<833xi32>> -> tensor<833xi32>
      %2 = flow.dispatch.tensor.load %arg4, offsets = [0, 0], sizes = [833, 833], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<833x833xf32>> -> tensor<833x833xf32>
      %3 = flow.dispatch.tensor.load %arg5, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
      %4 = tensor.empty() : tensor<f32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%1, %1, %2, %3 : tensor<833xi32>, tensor<833xi32>, tensor<833x833xf32>, tensor<f32>) outs(%4 : tensor<f32>) {
      ^bb0(%in: i32, %in_0: i32, %in_1: f32, %in_2: f32, %out: f32):
        %6 = arith.divf %in_1, %in_2 : f32
        %7 = arith.cmpi eq, %in, %in_0 : i32
        %8 = arith.select %7, %6, %cst : f32
        %9 = arith.addf %out, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %5, %arg6, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    util.return %0 : tensor<f32>
  }
}
// -----
module {
  util.func public @grouped_quantized_matmul(%arg0: tensor<4096x32x128xi4>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %0 = flow.tensor.reshape %arg2 : tensor<4096x32x1xf32> -> tensor<4096x32xf32>
    %1 = flow.tensor.reshape %arg3 : tensor<4096x32x1xf32> -> tensor<4096x32xf32>
    %2 = flow.tensor.reshape %arg1 : tensor<1x1x32x128xf32> -> tensor<32x128xf32>
    %3 = flow.dispatch.workgroups(%arg0, %0, %1, %2) : (tensor<4096x32x128xi4>, tensor<4096x32xf32>, tensor<4096x32xf32>, tensor<32x128xf32>) -> tensor<4096xf32> =
        (%arg4: !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>, %arg5: !flow.dispatch.tensor<readonly:tensor<4096x32xf32>>, %arg6: !flow.dispatch.tensor<readonly:tensor<4096x32xf32>>, %arg7: !flow.dispatch.tensor<readonly:tensor<32x128xf32>>, %arg8: !flow.dispatch.tensor<writeonly:tensor<4096xf32>>) {
      %cst = arith.constant 0.000000e+00 : f32
      %5 = flow.dispatch.tensor.load %arg4, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
      %6 = flow.dispatch.tensor.load %arg5, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32xf32>> -> tensor<4096x32xf32>
      %7 = flow.dispatch.tensor.load %arg6, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32xf32>> -> tensor<4096x32xf32>
      %8 = flow.dispatch.tensor.load %arg7, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf32>> -> tensor<32x128xf32>
      %9 = tensor.empty() : tensor<4096xf32>
      %10 = tensor.empty() : tensor<4096x32x128xf32>
      %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x32x128xi4>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%10 : tensor<4096x32x128xf32>) {
      ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
        %14 = arith.extui %in : i4 to i32
        %15 = arith.uitofp %14 : i32 to f32
        %16 = arith.subf %15, %in_1 : f32
        %17 = arith.mulf %16, %in_0 : f32
        linalg.yield %17 : f32
      } -> tensor<4096x32x128xf32>
      %12 = linalg.fill ins(%cst : f32) outs(%9 : tensor<4096xf32>) -> tensor<4096xf32>
      %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%8, %11 : tensor<32x128xf32>, tensor<4096x32x128xf32>) outs(%12 : tensor<4096xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %14 = arith.mulf %in, %in_0 : f32
        %15 = arith.addf %14, %out : f32
        linalg.yield %15 : f32
      } -> tensor<4096xf32>
      flow.dispatch.tensor.store %13, %arg8, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    %4 = flow.tensor.reshape %3 : tensor<4096xf32> -> tensor<1x1x4096xf32>
    util.return %4 : tensor<1x1x4096xf32>
  }
}
// -----
module {
  util.func public @verify_operand_cse(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view {
    %c12 = arith.constant 12 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %2 = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<?x12x?x64xf32>{%0, %1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[3] : index
    %5 = hal.tensor.import wait(%arg2) => %arg1 : !hal.buffer_view -> tensor<?x12x64x?xf32>{%3, %4}
    %6 = arith.maxui %0, %3 : index
    %7 = affine.apply affine_map<()[s0] -> (s0 * 12)>()[%0]
    %8 = flow.tensor.reshape %2 : tensor<?x12x?x64xf32>{%0, %1} -> tensor<?x?x64xf32>{%7, %1}
    %9 = affine.apply affine_map<()[s0] -> (s0 * 12)>()[%3]
    %10 = flow.tensor.reshape %5 : tensor<?x12x64x?xf32>{%3, %4} -> tensor<?x64x?xf32>{%9, %4}
    %11 = arith.muli %6, %c12 : index
    %12 = flow.dispatch.workgroups[%7, %1, %9, %4, %11](%8, %10, %7, %1, %9, %4, %11) : (tensor<?x?x64xf32>{%7, %1}, tensor<?x64x?xf32>{%9, %4}, index, index, index, index, index) -> tensor<?x?x?xf32>{%11, %1, %4} =
        (%arg4: !flow.dispatch.tensor<readonly:tensor<?x?x64xf32>>, %arg5: !flow.dispatch.tensor<readonly:tensor<?x64x?xf32>>, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>) {
      %17 = flow.dispatch.workload.ordinal %arg6, 0 : index
      %18 = flow.dispatch.workload.ordinal %arg7, 1 : index
      %19 = flow.dispatch.workload.ordinal %arg8, 2 : index
      %20 = flow.dispatch.workload.ordinal %arg9, 3 : index
      %21 = flow.dispatch.workload.ordinal %arg10, 4 : index
      %cst = arith.constant 0.000000e+00 : f32
      %22 = flow.dispatch.tensor.load %arg4, offsets = [0, 0, 0], sizes = [%17, %18, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x64xf32>>{%17, %18} -> tensor<?x?x64xf32>
      %23 = flow.dispatch.tensor.load %arg5, offsets = [0, 0, 0], sizes = [%19, 64, %20], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x64x?xf32>>{%19, %20} -> tensor<?x64x?xf32>
      %24 = tensor.empty(%21, %18, %20) : tensor<?x?x?xf32>
      %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %26 = linalg.batch_matmul ins(%22, %23 : tensor<?x?x64xf32>, tensor<?x64x?xf32>) outs(%25 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      flow.dispatch.tensor.store %26, %arg11, offsets = [0, 0, 0], sizes = [%21, %18, %20], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%21, %18, %20}
      flow.return
    } count(%arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg4, %arg5, %arg6, %arg7, %arg8
      flow.return %x, %y, %z : index, index, index
    }
    %13 = affine.apply affine_map<()[s0] -> (s0 floordiv 12)>()[%11]
    %14 = flow.tensor.reshape %12 : tensor<?x?x?xf32>{%11, %1, %4} -> tensor<?x12x?x?xf32>{%13, %1, %4}
    %15 = hal.tensor.barrier join(%14 : tensor<?x12x?x?xf32>) => %arg3 : !hal.fence
    %16 = hal.tensor.export %15 : tensor<?x12x?x?xf32>{%13, %1, %4} -> !hal.buffer_view
    util.return %16 : !hal.buffer_view
  }
}
