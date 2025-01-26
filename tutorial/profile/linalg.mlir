#map_1d_identity = affine_map<(m) -> (m)>

func.func @foo(
      %lhs : tensor<?xf32>,
      %rhs : tensor<?xf32>
    ) -> tensor<?xf32> {
  %c0f32 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %size = tensor.dim %lhs, %c0 : tensor<?xf32>
  %result_empty =  tensor.empty(%size) : tensor<?xf32>

  %result = linalg.generic {
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_identity
    ],
    iterator_types=["parallel"]
  } ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
    outs(%result_empty : tensor<?xf32>)
  {
    ^bb0(%lhs_entry : f32, %rhs_entry : f32, %unused_result_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  }
  -> tensor<?xf32>

  return %result : tensor<?xf32>
}
