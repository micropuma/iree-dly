util.func public @existing_count_region(%arg0 : index, %arg1 : index) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %0 = flow.dispatch.region[%arg0, %arg1] -> (tensor<?x?xf32>{%arg0, %arg1}) {
    %1 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
    flow.return %1 : tensor<?x?xf32>
  } count(%arg2 : index, %arg3 : index) -> (index, index, index) {
    flow.return %arg2, %arg3, %c1 : index, index, index
  }
  util.return %0 : tensor<?x?xf32>
}

util.func public @simple_test_with_cfg(%arg0: i1) -> (tensor<10x20xf32>) {
  %cst = arith.constant dense<1.000000e+00> : tensor<10x20xf32>
  %0 = flow.dispatch.region -> (tensor<10x20xf32>) {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<10x20xf32>
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = tensor.empty() : tensor<10x20xf32>
    flow.return %2 : tensor<10x20xf32>
  ^bb2:  // pred: ^bb0
    flow.return %cst_0 : tensor<10x20xf32>
  }
  util.return %0 : tensor<10x20xf32>
}
