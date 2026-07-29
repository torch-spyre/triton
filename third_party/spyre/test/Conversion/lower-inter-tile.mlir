// RUN: spyre-triton-opt %s --lower-inter-tile -split-input-file | FileCheck %s

// Tests for the --lower-inter-tile pass: tt.inter_tile_reduce → ktdp produce/reduce.

// Fold-away: W[axis] == 1 → partial forwarded, no KTDP ops.

// CHECK-LABEL: tt.func @fold_away_single_tile
// CHECK-NOT:     ktdp.inter_tile_produce
// CHECK-NOT:     ktdp.inter_tile_reduce
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         tt.return %arg0
tt.func @fold_away_single_tile(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 1 : i64},
          coreIdToWkSlice = [{x = 0 : i64}]}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}

// -----

// W[y]==1 on non-innermost axis also folds away.

// CHECK-LABEL: tt.func @fold_away_non_innermost_axis
// CHECK-NOT:     ktdp.inter_tile_produce
// CHECK-NOT:     ktdp.inter_tile_reduce
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         tt.return %arg0
tt.func @fold_away_non_innermost_axis(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "y" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 4 : i64, y = 1 : i64},
          coreIdToWkSlice = [{x = 0 : i64, y = 0 : i64}, {x = 1 : i64, y = 0 : i64},
                             {x = 2 : i64, y = 0 : i64}, {x = 3 : i64, y = 0 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// all_reduce: produce/reduce pair emitted, tt.inter_tile_reduce erased.

// CHECK-LABEL: tt.func @all_reduce_basic
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         %[[FUTURE:.*]] = ktdp.inter_tile_produce
// CHECK:           ktdp.yield_partial %arg0
// CHECK:         ktdp.inter_tile_reduce(%[[FUTURE]])
// CHECK:           linalg.add
// CHECK:           ktdp.yield_reduced
// CHECK:         tt.return
tt.func @all_reduce_basic(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 4 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}, {x = 2 : i64}, {x = 3 : i64}]}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}

// -----

// all_reduce: consumer_tiles_per_group == producer_tiles_per_group (same set).

// CHECK-LABEL: tt.func @all_reduce_consumer_equals_producer
// CHECK:         ktdp.inter_tile_produce producer_tiles_per_group = #[[$PROD:.*]] ->
// CHECK:         ktdp.inter_tile_reduce({{.*}}) consumer_tiles_per_group = #[[$PROD]],
tt.func @all_reduce_consumer_equals_producer(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// reduce_to_one: consumer set is pick0 (single-point equality).

// CHECK-LABEL: tt.func @reduce_to_one_basic
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce({{.*}}) consumer_tiles_per_group = #[[$PICK0:.*]],
// CHECK:           linalg.add
// CHECK:           ktdp.yield_reduced
tt.func @reduce_to_one_basic(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "reduce_to_one" combiner = "add"
         {numWkSlicesPerDim = {x = 4 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}, {x = 2 : i64}, {x = 3 : i64}]}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}

// -----

// pick0 selects tile with axis_value==0, not necessarily tile-id 0.
// core_map=[{x:1},{x:0}]: tile 1 has x=0 → consumer set encodes d0 - 1 == 0.

// CHECK: #[[$PICK0:.*]] = affine_set<(d0)[s0] : (d0 - 1 == 0)>
// CHECK-LABEL: tt.func @reduce_to_one_pick0_not_lowest_tile
// CHECK:         ktdp.inter_tile_reduce({{.*}}) consumer_tiles_per_group = #[[$PICK0]],
tt.func @reduce_to_one_pick0_not_lowest_tile(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "reduce_to_one" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 1 : i64}, {x = 0 : i64}]}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}

// -----

// reduce_to_one with valid depWkSlices is accepted.

// CHECK-LABEL: tt.func @reduce_to_one_with_dep
// CHECK:         ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce
tt.func @reduce_to_one_with_dep(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "reduce_to_one" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}],
          depWkSlices = {"0" = [0 : i64, 1 : i64]}}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}

// -----

// Combiners: shorthand name → corresponding linalg op in reduce region.

// CHECK-LABEL: tt.func @combiner_add
// CHECK:         linalg.add
// CHECK-SAME:      outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
tt.func @combiner_add(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: tt.func @combiner_max
// CHECK:         linalg.max
// CHECK-SAME:      outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
tt.func @combiner_max(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "max"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: tt.func @combiner_mul
// CHECK:         linalg.mul
// CHECK-SAME:      outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
tt.func @combiner_mul(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "mul"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Result type preserved: f16 partial → f16 result.

// CHECK-LABEL: tt.func @result_type_f16
// CHECK:         ktdp.inter_tile_reduce({{.*}}){{.*}}-> tensor<8xf16>
tt.func @result_type_f16(%p: tensor<8xf16>, %id: tensor<8xf16>) -> tensor<8xf16> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf16>)
         identities(%id : tensor<8xf16>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf16>)
  tt.return %0 : tensor<8xf16>
}

// -----

// Leading dim 1 is preserved (no silent rank reduction).

// CHECK-LABEL: tt.func @result_type_preserves_rank
// CHECK:         ktdp.inter_tile_reduce({{.*}}){{.*}}-> tensor<1x16xf32>
tt.func @result_type_preserves_rank(%p: tensor<1x16xf32>, %id: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<1x16xf32>)
         identities(%id : tensor<1x16xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 4 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}, {x = 2 : i64}, {x = 3 : i64}]}
         -> (tensor<1x16xf32>)
  tt.return %0 : tensor<1x16xf32>
}

// -----

// Multi-arity: 2 partials → 2 results.

// CHECK-LABEL: tt.func @result_type_multi_arity
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce
tt.func @result_type_multi_arity(%p0: tensor<8xf32>, %p1: tensor<8xf32>,
                                  %id0: tensor<8xf32>, %id1: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>) {
  %0, %1 = tt.inter_tile_reduce
              partials(%p0, %p1 : tensor<8xf32>, tensor<8xf32>)
              identities(%id0, %id1 : tensor<8xf32>, tensor<8xf32>)
              axis = "x" mode = "all_reduce" combiner = "add"
              {numWkSlicesPerDim = {x = 2 : i64},
               coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
              -> (tensor<8xf32>, tensor<8xf32>)
  tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

// Multi-axis: 2 groups x 2 tiles (axis="x" reduction, "y" group key).

// CHECK-LABEL: tt.func @multi_axis_two_groups
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce
tt.func @multi_axis_two_groups(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {y = 2 : i64, x = 2 : i64},
          coreIdToWkSlice = [{y = 0 : i64, x = 0 : i64}, {y = 0 : i64, x = 1 : i64},
                             {y = 1 : i64, x = 0 : i64}, {y = 1 : i64, x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Double all_reduce: two independent reduces lower to 2 produce + 2 reduce.

// CHECK-LABEL: tt.func @double_all_reduce
// CHECK-NOT:     tt.inter_tile_reduce
// CHECK:         ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce
// CHECK:         linalg.max
// CHECK:         ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce
// CHECK:         linalg.add
tt.func @double_all_reduce(%pmax: tensor<8xf32>, %psum: tensor<8xf32>,
                            %idmax: tensor<8xf32>, %idsum: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>) {
  %rowmax = tt.inter_tile_reduce
              partials(%pmax : tensor<8xf32>)
              identities(%idmax : tensor<8xf32>)
              axis = "out" mode = "all_reduce" combiner = "max"
              {numWkSlicesPerDim = {mb = 2 : i64, out = 2 : i64},
               coreIdToWkSlice = [{mb = 0 : i64, out = 0 : i64}, {mb = 0 : i64, out = 1 : i64},
                                  {mb = 1 : i64, out = 0 : i64}, {mb = 1 : i64, out = 1 : i64}]}
              -> (tensor<8xf32>)
  %rowsum = tt.inter_tile_reduce
              partials(%psum : tensor<8xf32>)
              identities(%idsum : tensor<8xf32>)
              axis = "out" mode = "all_reduce" combiner = "add"
              {numWkSlicesPerDim = {mb = 2 : i64, out = 2 : i64},
               coreIdToWkSlice = [{mb = 0 : i64, out = 0 : i64}, {mb = 0 : i64, out = 1 : i64},
                                  {mb = 1 : i64, out = 0 : i64}, {mb = 1 : i64, out = 1 : i64}]}
              -> (tensor<8xf32>)
  tt.return %rowmax, %rowsum : tensor<8xf32>, tensor<8xf32>
}

// -----

// Affine set shapes: producer_tiles_per_group is (d0)[s0] with 2 constraints,
// groups is (d0) with 2 constraints. Single-axis, gsize=4, ngroups=1.

// CHECK:       #[[$PROD_4:.*]] = affine_set<(d0)[s0] : (d0 - s0 * 4 >= 0, -d0 + s0 * 4 + 3 >= 0)>
// CHECK:       #[[$GROUPS_1:.*]] = affine_set<(d0) : (d0 >= 0, -d0 >= 0)>
// CHECK-LABEL: tt.func @groups_partition_single_axis
// CHECK:         ktdp.inter_tile_produce producer_tiles_per_group = #[[$PROD_4]] -> <(tensor<16xf32>), groups = #[[$GROUPS_1]]>
// CHECK:         ktdp.inter_tile_reduce({{.*}}) consumer_tiles_per_group = #[[$PROD_4]],{{.*}}: <(tensor<16xf32>), groups = #[[$GROUPS_1]]>
tt.func @groups_partition_single_axis(%p: tensor<16xf32>, %id: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<16xf32>)
         identities(%id : tensor<16xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 4 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}, {x = 2 : i64}, {x = 3 : i64}]}
         -> (tensor<16xf32>)
  tt.return %0 : tensor<16xf32>
}

// -----

// Multi-group affine set shapes: gsize=2, ngroups=2.
// producer_tiles_per_group: (d0)[s0] with 2 constraints; groups: (d0) with 2 constraints.

// CHECK:       #[[$PROD_2:.*]] = affine_set<(d0)[s0] : (d0 - s0 * 2 >= 0, -d0 + s0 * 2 + 1 >= 0)>
// CHECK:       #[[$GROUPS_2:.*]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK-LABEL: tt.func @groups_partition_multi_group
// CHECK:         ktdp.inter_tile_produce producer_tiles_per_group = #[[$PROD_2]] -> <(tensor<8xf32>), groups = #[[$GROUPS_2]]>
// CHECK:         ktdp.inter_tile_reduce({{.*}}) consumer_tiles_per_group = #[[$PROD_2]],{{.*}}: <(tensor<8xf32>), groups = #[[$GROUPS_2]]>
tt.func @groups_partition_multi_group(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {y = 2 : i64, x = 2 : i64},
          coreIdToWkSlice = [{y = 0 : i64, x = 0 : i64}, {y = 0 : i64, x = 1 : i64},
                             {y = 1 : i64, x = 0 : i64}, {y = 1 : i64, x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// Future single-use: produce has exactly one consumer (the reduce op).

// CHECK-LABEL: tt.func @future_single_use
// CHECK:         %[[F:.*]] = ktdp.inter_tile_produce
// CHECK:         ktdp.inter_tile_reduce(%[[F]])
// CHECK-NOT:     ktdp.inter_tile_reduce(%[[F]])
tt.func @future_single_use(%p: tensor<8xf32>, %id: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<8xf32>)
         identities(%id : tensor<8xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<8xf32>)
  tt.return %0 : tensor<8xf32>
}

// -----

// No-op: module without tt.inter_tile_reduce passes through unchanged.

// CHECK-LABEL: tt.func @no_op
// CHECK-NOT:     ktdp.inter_tile_produce
// CHECK-NOT:     ktdp.inter_tile_reduce
// CHECK:         tt.return %arg0
tt.func @no_op(%a: tensor<16xf32>) -> tensor<16xf32> {
  tt.return %a : tensor<16xf32>
}
