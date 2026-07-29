// RUN: spyre-triton-opt %s --lower-inter-tile -split-input-file -verify-diagnostics

// Negative tests for --lower-inter-tile: precondition and validation diagnostics.

tt.func @missing_work_slice_attrs(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{'tt.inter_tile_reduce' op requires attribute 'coreIdToWkSlice'}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @unknown_axis(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{axis 'x' not in numWkSlicesPerDim}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {y = 2 : i64},
          coreIdToWkSlice = [{y = 0 : i64}, {y = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @unknown_mode(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{unknown mode 'bad_mode'}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "bad_mode" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @broadcast_rejected(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{mode 'broadcast' is not yet supported}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "broadcast" combiner = ""
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @reduce_scatter_rejected(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{mode 'reduce_scatter' is not yet supported}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "reduce_scatter" combiner = "add"
         scatter_dimension = 0
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @scatter_dim_without_reduce_scatter(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{scatter_dimension only valid for reduce_scatter}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         scatter_dimension = 0
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @custom_combiner_rejected(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{custom combiner regions are not yet supported}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = ""
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

tt.func @dep_invalid_consumer(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error @+1 {{depWkSlices key 1 is not a valid consumer for mode 'reduce_to_one' (only indices [0, 1) are consumers)}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "reduce_to_one" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}],
          depWkSlices = {"1" = [0 : i64, 1 : i64]}}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}
