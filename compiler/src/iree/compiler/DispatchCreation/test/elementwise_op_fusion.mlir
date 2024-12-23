// RUN: iree-opt --iree-dispatch-creation-elementwise-op-fusion --split-input-file --mlir-print-local-scope  %s | FileCheck %s

util.func public @transpose_attention(%arg0: tensor<4x64x32x128xf16>, %arg1: tensor<4x64x32x128xf16>, %arg2: tensor<4x64x32x128xf16>, %arg3: f16) -> tensor<4x64x4096xf16> {
  %0 = tensor.empty() : tensor<4x32x64x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %4 = tensor.empty() : tensor<4x32x64x128xf16>
  %5 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%1, %2, %3, %arg3 : tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, f16) outs(%4 : tensor<4x32x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<4x32x64x128xf16>
  %6 = tensor.empty() : tensor<4x64x32x128xf16>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<4x32x64x128xf16>) outs(%6 : tensor<4x64x32x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x64x32x128xf16>
  %collapsed = tensor.collapse_shape %7 [[0], [1], [2, 3]] : tensor<4x64x32x128xf16> into tensor<4x64x4096xf16>
  util.return %collapsed : tensor<4x64x4096xf16>
}

// CHECK-LABEL: util.func public @transpose_attention
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: f16
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d5)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]

// -----

util.func public @transposed_attention_masked(%arg0: tensor<4x64x32x128xf16>, %arg1: tensor<4x64x32x128xf16>, %arg2: tensor<4x64x32x128xf16>, %arg3: f16, %arg4: tensor<4x64x32x64xf16>) -> tensor<4x64x4096xf16> {
  %0 = tensor.empty() : tensor<4x32x64x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %empty = tensor.empty() : tensor<4x32x64x64xf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<4x64x32x64xf16>) outs(%empty : tensor<4x32x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x64xf16>
  %5 = tensor.empty() : tensor<4x32x64x128xf16>
  %6 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%1, %2, %3, %arg3, %4 : tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, f16, tensor<4x32x64x64xf16>) outs(%5 : tensor<4x32x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<4x32x64x128xf16>
  %7 = tensor.empty() : tensor<4x64x32x128xf16>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<4x32x64x128xf16>) outs(%7 : tensor<4x64x32x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x64x32x128xf16>
  %collapsed = tensor.collapse_shape %8 [[0], [1], [2, 3]] : tensor<4x64x32x128xf16> into tensor<4x64x4096xf16>
  util.return %collapsed : tensor<4x64x4096xf16>
}

// CHECK-LABEL: util.func public @transposed_attention_masked
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: f16
//  CHECK-SAME:   %[[ARG4:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d5)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]

// -----

util.func public @transpose_matmul(%arg0 : tensor<100x100xf16>, %arg1 : tensor<100x100xf16>) -> (tensor<100x100xf16>) {
  %0 = tensor.empty() : tensor<100x100xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<100x100xf16>) outs(%0 : tensor<100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<100x100xf16>
  %2 = tensor.empty() : tensor<100x100xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<100x100xf16>) outs(%2 : tensor<100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<100x100xf16>
  %5 = tensor.empty() : tensor<100x100xf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %3: tensor<100x100xf16>, tensor<100x100xf16>) outs(%5 : tensor<100x100xf16>) {
  ^bb0(%in: f16, %in_0 : f16, %out: f16):
    %01 = arith.mulf %in, %in_0 : f16
    %02 = arith.addf %01, %out: f16
    linalg.yield %02  : f16
  } -> tensor<100x100xf16>
  util.return  %4 : tensor<100x100xf16>
}

// CHECK-LABEL: util.func public @transpose_matmul
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[RET:.+]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d0)>
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]]

// -----

util.func public @fuse_generic_gather(
  %11 :tensor<128256x4096xf16>, %12 : tensor<4x?xi64>,
  %13 : tensor<4x?x4096xf32>, %14 : tensor<128256x4096xf32>)
    -> tensor<4x?x4096xf32>{

  %15 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%11 : tensor<128256x4096xf16>)
    outs(%14 : tensor<128256x4096xf32>) {
      ^bb0(%in: f16, %out: f32):
        %17 = arith.extf %in : f16 to f32
        linalg.yield %17 : f32
    } -> tensor<128256x4096xf32>
  %16 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%12 : tensor<4x?xi64>)
    outs(%13 : tensor<4x?x4096xf32>) {
      ^bb0(%in: i64, %out: f32):
        %17 = arith.index_cast %in : i64 to index
        %18 = linalg.index 2 : index
        %extracted = tensor.extract %15[%17, %18] : tensor<128256x4096xf32>
        linalg.yield %extracted : f32
      } -> tensor<4x?x4096xf32>
  util.return %16 : tensor<4x?x4096xf32>
}

// CHECK:         %[[INDEX0:[a-zA-Z0-9]+]] = arith.index_cast %in : i64 to index
// CHECK:         %[[INDEX1:[a-zA-Z0-9]+]] = linalg.index 2 : index
// CHECK-NEXT:    %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR0:.+]][%[[INDEX0]], %[[INDEX1]]] : tensor<128256x4096xf16>
// CHECK-NEXT:    %[[RES:[a-zA-Z0-9]+]] = arith.extf %[[EXTRACTED]] : f16 to f32
// CHECK-NEXT:    linalg.yield %[[RES]] : f32


// -----

util.func public @fuse_generic_gather2(
  %11 :tensor<128256x4096xf16>, %12 : tensor<4x?xi64>,
  %13 : tensor<4x?x4096xf32>, %14 : tensor<128256x4096xf32>)
    -> tensor<4x?x4096xf32>{

  %15 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%11 : tensor<128256x4096xf16>)
    outs(%14 : tensor<128256x4096xf32>) {
      ^bb0(%in: f16, %out: f32):
        %17 = arith.extf %in : f16 to f32
        linalg.yield %17 : f32
    } -> tensor<128256x4096xf32>
  %16 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%12 : tensor<4x?xi64>)
    outs(%13 : tensor<4x?x4096xf32>) {
      ^bb0(%in: i64, %out: f32):
        %17 = arith.index_cast %in : i64 to index
        %18 = linalg.index 2 : index
        %extracted = tensor.extract %15[%17, %18] : tensor<128256x4096xf32>
        %result = arith.addf %extracted, %extracted : f32
        %result2 = arith.mulf %extracted, %extracted : f32
        %final = arith.addf %result, %result2 : f32
        linalg.yield %final: f32
      } -> tensor<4x?x4096xf32>
  util.return %16 : tensor<4x?x4096xf32>
}

// CHECK:         %[[INDEX0:[a-zA-Z0-9]+]] = arith.index_cast %in : i64 to index
// CHECK:         %[[INDEX1:[a-zA-Z0-9]+]] = linalg.index 2 : index
// CHECK-NEXT:    %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR0:.+]][%[[INDEX0]], %[[INDEX1]]] : tensor<128256x4096xf16>
// CHECK-NEXT:    %[[RES:[a-zA-Z0-9]+]] = arith.extf %[[EXTRACTED]] : f16 to f32
// CHECK-NEXT:    %[[RES2:[a-zA-Z0-9]+]] = arith.addf %[[RES]], %[[RES]] : f32
// CHECK-NEXT:    %[[RES3:[a-zA-Z0-9]+]] = arith.mulf %[[RES]], %[[RES]] : f32
// CHECK-NEXT:    %[[RES4:[a-zA-Z0-9]+]] = arith.addf %[[RES2]], %[[RES3]] : f32
// CHECK-NEXT:    linalg.yield %[[RES4]] : f32
