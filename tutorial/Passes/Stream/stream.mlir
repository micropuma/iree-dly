#map = affine_map<(d0) -> (d0)>
module {
  util.global private @_constant : !stream.resource<constant>
  util.global private @_constant__size : index
  util.initializer {
    %cst = stream.tensor.constant : tensor<10xf32> in !stream.resource<constant> = dense<[0.000000e+00, 0.00999999977, 2.000000e-02, 3.000000e-02, 4.000000e-02, 5.000000e-02, 6.000000e-02, 7.000000e-02, 8.000000e-02, 9.000000e-02]> : tensor<10xf32>
    %0 = stream.resource.size %cst : !stream.resource<constant>
    util.global.store %cst, @_constant : !stream.resource<constant>
    util.global.store %0, @_constant__size : index
    util.return
  }
  stream.executable private @test_dispatch_0 {
    stream.executable.export public @test_dispatch_0_generic_10 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @test_dispatch_0_generic_10(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<10xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<10xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<10xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<10xf32>> -> tensor<10xf32>
        %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : tensor<10xf32>, tensor<10xf32>) outs(%5 : tensor<10xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.addf %in, %in_0 : f32
          linalg.yield %7 : f32
        } -> tensor<10xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<readwrite:tensor<10xf32>>
        return
      }
    }
  }
  func.func @test(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c10 = arith.constant 10 : index
    %_constant = util.global.load @_constant : !stream.resource<constant>
    %_constant__size = util.global.load @_constant__size : index
    %0 = stream.async.transfer %_constant : !stream.resource<constant>{%_constant__size} -> !stream.resource<*>{%_constant__size}
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    %c1 = arith.constant 1 : index
    %c10_0 = arith.constant 10 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("tensor") shape([%c1, %c10_0]) type(%element_type_f32) encoding(%dense_row_major)
    %1 = stream.tensor.sizeof tensor<1x10xf32> : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<1x10xf32> in !stream.resource<external>{%1}
    %3 = stream.async.transfer %2 : !stream.resource<external>{%1} -> !stream.resource<*>{%1}
    %4 = stream.tensor.sizeof tensor<10xf32> : index
    %5 = stream.tensor.clone %3 : tensor<1x10xf32> in !stream.resource<*>{%1} -> tensor<10xf32> in !stream.resource<*>{%4}
    %6 = stream.tensor.sizeof tensor<10xf32> : index
    %empty = stream.tensor.empty : tensor<10xf32> in !stream.resource<*>{%6}
    %c0 = arith.constant 0 : index
    %7 = stream.async.dispatch @test_dispatch_0::@test_dispatch_0_generic_10[%c10](%5[%c0 to %4 for %4], %0[%c0 to %_constant__size for %_constant__size], %empty[%c0 to %6 for %6]) : (!stream.resource<*>{%4}, !stream.resource<*>{%_constant__size}, !stream.resource<*>{%6}) -> %empty{%6}
    %8 = stream.tensor.sizeof tensor<1x10xf32> : index
    %9 = stream.tensor.clone %7 : tensor<10xf32> in !stream.resource<*>{%6} -> tensor<1x10xf32> in !stream.resource<*>{%8}
    %10 = stream.async.transfer %9 : !stream.resource<*>{%8} -> !stream.resource<external>{%8}
    %11 = stream.tensor.export %10 : tensor<1x10xf32> in !stream.resource<external>{%8} -> !hal.buffer_view
    return %11 : !hal.buffer_view
  }
}

