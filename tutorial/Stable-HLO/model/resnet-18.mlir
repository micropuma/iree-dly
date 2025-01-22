module attributes {torch.debug_module_name = "ResNet"} {
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = stablehlo.constant dense_resource<__elided__> : tensor<1000xf32>
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<1000x512xf32>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_6 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_8 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_10 = stablehlo.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_11 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_12 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_13 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<512x256x1x1xf32>
    %cst_15 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_16 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_17 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_18 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_19 = stablehlo.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_20 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_22 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<512x256x3x3xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<256x128x1x1xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_43 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_48 = stablehlo.constant dense_resource<__elided__> : tensor<256x128x3x3xf32>
    %cst_49 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_50 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_51 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_52 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_53 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_54 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_55 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_56 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_57 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_58 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_59 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_60 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_61 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_62 = stablehlo.constant dense_resource<__elided__> : tensor<128x64x1x1xf32>
    %cst_63 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_64 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_65 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_66 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_67 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_68 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_70 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_72 = stablehlo.constant dense_resource<__elided__> : tensor<128x64x3x3xf32>
    %cst_73 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_74 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_75 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_76 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_77 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_78 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_79 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_80 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_81 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_82 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_83 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_84 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_85 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_86 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_87 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_88 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_89 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_90 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_91 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_92 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_93 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_94 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_95 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_96 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_97 = stablehlo.constant dense_resource<__elided__> : tensor<64x3x7x7xf32>
    %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x112x112xf32>
    %cst_99 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<49> : tensor<i64>
    %0 = stablehlo.convolution(%arg0, %cst_97) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %1 = "stablehlo.batch_norm_inference"(%0, %cst_94, %cst_93, %cst_96, %cst_95) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %2 = stablehlo.maximum %1, %cst_98 : tensor<1x64x112x112xf32>
    %3 = "stablehlo.reduce_window"(%2, %cst_99) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %77 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %77 : tensor<f32>
    }) : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
    %4 = stablehlo.convolution(%3, %cst_92) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %5 = "stablehlo.batch_norm_inference"(%4, %cst_89, %cst_88, %cst_91, %cst_90) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.maximum %5, %cst_100 : tensor<1x64x56x56xf32>
    %7 = stablehlo.convolution(%6, %cst_87) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %8 = "stablehlo.batch_norm_inference"(%7, %cst_84, %cst_83, %cst_86, %cst_85) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %9 = stablehlo.add %8, %3 : tensor<1x64x56x56xf32>
    %10 = stablehlo.maximum %9, %cst_100 : tensor<1x64x56x56xf32>
    %11 = stablehlo.convolution(%10, %cst_82) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %12 = "stablehlo.batch_norm_inference"(%11, %cst_79, %cst_78, %cst_81, %cst_80) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %13 = stablehlo.maximum %12, %cst_100 : tensor<1x64x56x56xf32>
    %14 = stablehlo.convolution(%13, %cst_77) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %15 = "stablehlo.batch_norm_inference"(%14, %cst_74, %cst_73, %cst_76, %cst_75) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %16 = stablehlo.add %15, %10 : tensor<1x64x56x56xf32>
    %17 = stablehlo.maximum %16, %cst_100 : tensor<1x64x56x56xf32>
    %18 = stablehlo.convolution(%17, %cst_72) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x28x28xf32>
    %19 = "stablehlo.batch_norm_inference"(%18, %cst_69, %cst_68, %cst_71, %cst_70) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %20 = stablehlo.maximum %19, %cst_101 : tensor<1x128x28x28xf32>
    %21 = stablehlo.convolution(%20, %cst_67) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %22 = "stablehlo.batch_norm_inference"(%21, %cst_64, %cst_63, %cst_66, %cst_65) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %23 = stablehlo.convolution(%17, %cst_62) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<1x128x28x28xf32>
    %24 = "stablehlo.batch_norm_inference"(%23, %cst_59, %cst_63, %cst_61, %cst_60) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %25 = stablehlo.add %22, %24 : tensor<1x128x28x28xf32>
    %26 = stablehlo.maximum %25, %cst_101 : tensor<1x128x28x28xf32>
    %27 = stablehlo.convolution(%26, %cst_58) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %28 = "stablehlo.batch_norm_inference"(%27, %cst_55, %cst_54, %cst_57, %cst_56) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %29 = stablehlo.maximum %28, %cst_101 : tensor<1x128x28x28xf32>
    %30 = stablehlo.convolution(%29, %cst_53) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %31 = "stablehlo.batch_norm_inference"(%30, %cst_50, %cst_49, %cst_52, %cst_51) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %32 = stablehlo.add %31, %26 : tensor<1x128x28x28xf32>
    %33 = stablehlo.maximum %32, %cst_101 : tensor<1x128x28x28xf32>
    %34 = stablehlo.convolution(%33, %cst_48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<1x256x14x14xf32>
    %35 = "stablehlo.batch_norm_inference"(%34, %cst_45, %cst_44, %cst_47, %cst_46) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %36 = stablehlo.maximum %35, %cst_102 : tensor<1x256x14x14xf32>
    %37 = stablehlo.convolution(%36, %cst_43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %38 = "stablehlo.batch_norm_inference"(%37, %cst_40, %cst_39, %cst_42, %cst_41) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %39 = stablehlo.convolution(%33, %cst_38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<1x256x14x14xf32>
    %40 = "stablehlo.batch_norm_inference"(%39, %cst_35, %cst_39, %cst_37, %cst_36) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %41 = stablehlo.add %38, %40 : tensor<1x256x14x14xf32>
    %42 = stablehlo.maximum %41, %cst_102 : tensor<1x256x14x14xf32>
    %43 = stablehlo.convolution(%42, %cst_34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %44 = "stablehlo.batch_norm_inference"(%43, %cst_31, %cst_30, %cst_33, %cst_32) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %45 = stablehlo.maximum %44, %cst_102 : tensor<1x256x14x14xf32>
    %46 = stablehlo.convolution(%45, %cst_29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %47 = "stablehlo.batch_norm_inference"(%46, %cst_26, %cst_25, %cst_28, %cst_27) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %48 = stablehlo.add %47, %42 : tensor<1x256x14x14xf32>
    %49 = stablehlo.maximum %48, %cst_102 : tensor<1x256x14x14xf32>
    %50 = stablehlo.convolution(%49, %cst_24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<1x512x7x7xf32>
    %51 = "stablehlo.batch_norm_inference"(%50, %cst_21, %cst_20, %cst_23, %cst_22) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %52 = stablehlo.maximum %51, %cst_103 : tensor<1x512x7x7xf32>
    %53 = stablehlo.convolution(%52, %cst_19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %54 = "stablehlo.batch_norm_inference"(%53, %cst_16, %cst_15, %cst_18, %cst_17) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %55 = stablehlo.convolution(%49, %cst_14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<1x512x7x7xf32>
    %56 = "stablehlo.batch_norm_inference"(%55, %cst_11, %cst_15, %cst_13, %cst_12) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %57 = stablehlo.add %54, %56 : tensor<1x512x7x7xf32>
    %58 = stablehlo.maximum %57, %cst_103 : tensor<1x512x7x7xf32>
    %59 = stablehlo.convolution(%58, %cst_10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %60 = "stablehlo.batch_norm_inference"(%59, %cst_7, %cst_6, %cst_9, %cst_8) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %61 = stablehlo.maximum %60, %cst_103 : tensor<1x512x7x7xf32>
    %62 = stablehlo.convolution(%61, %cst_5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %63 = "stablehlo.batch_norm_inference"(%62, %cst_2, %cst_1, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %64 = stablehlo.add %63, %58 : tensor<1x512x7x7xf32>
    %65 = stablehlo.maximum %64, %cst_103 : tensor<1x512x7x7xf32>
    %66 = "stablehlo.reduce_window"(%65, %cst_104) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 7, 7>, window_strides = array<i64: 1, 1, 7, 7>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %77 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %77 : tensor<f32>
    }) : (tensor<1x512x7x7xf32>, tensor<f32>) -> tensor<1x512x1x1xf32>
    %67 = stablehlo.convert %c : (tensor<i64>) -> tensor<f32>
    %68 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %69 = stablehlo.broadcast_in_dim %67, dims = [] : (tensor<f32>) -> tensor<1x512x1x1xf32>
    %70 = stablehlo.divide %68, %69 : tensor<1x512x1x1xf32>
    %71 = stablehlo.reshape %70 : (tensor<1x512x1x1xf32>) -> tensor<1x512xf32>
    %72 = stablehlo.transpose %cst_0, dims = [1, 0] : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
    %73 = stablehlo.dot_general %71, %72, contracting_dims = [1] x [0] : (tensor<1x512xf32>, tensor<512x1000xf32>) -> tensor<1x1000xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %75 = stablehlo.broadcast_in_dim %cst, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x1000xf32>
    return %76 : tensor<1x1000xf32>
  }
}

