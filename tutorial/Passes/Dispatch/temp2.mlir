module {
  util.func public @existing_count_region(%arg0: index, %arg1: index) -> tensor<?x?xf32> {
    %c1 = arith.constant 1 : index
    %0 = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1, %arg0, %arg1) : (index, index, index, index) -> tensor<?x?xf32>{%arg0, %arg1} =
        (%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %1 = tensor.empty(%arg4, %arg5) : tensor<?x?xf32>
      flow.dispatch.tensor.store %1, %arg6, offsets = [0, 0], sizes = [%arg4, %arg5], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg4, %arg5}
      flow.return
    } count(%arg2: index, %arg3: index) -> (index, index, index) {
      %c1_0 = arith.constant 1 : index
      flow.return %arg2, %arg3, %c1_0 : index, index, index
    }
    util.return %0 : tensor<?x?xf32>
  }
  util.func public @simple_test_with_cfg(%arg0: i1) -> tensor<10x20xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<10x20xf32>
    %0 = flow.dispatch.workgroups(%arg0) : (i1) -> tensor<10x20xf32> =
        (%arg1: i1, %arg2: !flow.dispatch.tensor<writeonly:tensor<10x20xf32>>) {
      %cst_0 = arith.constant dense<1.000000e+00> : tensor<10x20xf32>
      cf.cond_br %arg1, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %1 = tensor.empty() : tensor<10x20xf32>
      flow.dispatch.tensor.store %1, %arg2, offsets = [0, 0], sizes = [10, 20], strides = [1, 1] : tensor<10x20xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x20xf32>>
      flow.return
    ^bb2:  // pred: ^bb0
      flow.dispatch.tensor.store %cst_0, %arg2, offsets = [0, 0], sizes = [10, 20], strides = [1, 1] : tensor<10x20xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x20xf32>>
      flow.return
    }
    util.return %0 : tensor<10x20xf32>
  }
}

