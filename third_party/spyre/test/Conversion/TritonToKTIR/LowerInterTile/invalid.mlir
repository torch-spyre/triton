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

// -----

// Group derivation: numTiles must divide evenly by gsize = W[axis].

tt.func @tile_count_indivisible(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{tile count 3 does not divide evenly by gsize=2 for axis 'x'}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}, {x = 0 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Group derivation: coreIdToWkSlice must not be empty.

tt.func @core_map_empty(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{coreIdToWkSlice is empty}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = []}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Group derivation: every coreIdToWkSlice entry must carry the reduction axis.

tt.func @core_map_entry_missing_axis(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{coreIdToWkSlice entry 1 has no key 'x'}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {y = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Group derivation: the non-axis tuples must number exactly numTiles/gsize.
// 4 tiles, gsize=2 → 2 groups expected, but every tile has a distinct y.

tt.func @group_count_mismatch(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{expected 2 groups (numTiles/W[axis]=4/2) but found 4 distinct non-axis tuples}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {y = 4 : i64, x = 2 : i64},
          coreIdToWkSlice = [{y = 0 : i64, x = 0 : i64}, {y = 1 : i64, x = 0 : i64},
                             {y = 2 : i64, x = 0 : i64}, {y = 3 : i64, x = 0 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Group derivation: members of a group must be contiguous tile ids.
// y is the group key, so groups are {0,2} and {1,3} — not contiguous.

tt.func @group_not_contiguous(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{group 0 is not contiguous: expected tile 1 at position 1, got 2 (non-contiguous groups not yet supported)}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {y = 2 : i64, x = 2 : i64},
          coreIdToWkSlice = [{y = 0 : i64, x = 0 : i64}, {y = 1 : i64, x = 0 : i64},
                             {y = 0 : i64, x = 1 : i64}, {y = 1 : i64, x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// pick0 derivation: each group needs exactly one tile at axis == 0 — none here.

tt.func @group_without_axis_zero(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{group 0 has no tile with x=0}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 1 : i64}, {x = 2 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// pick0 derivation: ... and not more than one.

tt.func @group_with_two_axis_zero(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{group 0 has more than one tile with x=0}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 0 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// reduce_to_one: the pick0 tile ids must form an arithmetic sequence, since the
// consumer set is the single affine equality i == base + g*stride.
// 3 groups whose pick0 tiles are 0, 3, 4 — stride 3 then 1.

tt.func @pick0_not_arithmetic(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{reduce_to_one: pick0 tile-ids are not an arithmetic sequence (non-uniform pick0 layouts are not yet supported)}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "reduce_to_one" combiner = "add"
         {numWkSlicesPerDim = {y = 3 : i64, x = 2 : i64},
          coreIdToWkSlice = [{y = 0 : i64, x = 0 : i64}, {y = 0 : i64, x = 1 : i64},
                             {y = 1 : i64, x = 1 : i64}, {y = 1 : i64, x = 0 : i64},
                             {y = 2 : i64, x = 0 : i64}, {y = 2 : i64, x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Combiner shorthand: only add/max/mul are known.

tt.func @unknown_combiner(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{unknown shorthand combiner 'min'}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "min"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Missing numWkSlicesPerDim alone (the op verifier catches it before the pass).

tt.func @missing_num_wk_slices(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{'tt.inter_tile_reduce' op requires attribute 'numWkSlicesPerDim'}}
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}
