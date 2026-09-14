// RUN: spyre-triton-opt %s --hbm-roundtrip -split-input-file -verify-diagnostics | FileCheck %s

// Two chained computes: the exp result goes to a spill buffer and is read back
// before the sqrt, so the two become separate compute groups. The buffer is a new
// `index` argument (%arg2) and is described on the module for the launcher.
//
// The `outs` of both generics is one `linalg.fill` over one `tensor.empty` --
// which is what canonicalize and CSE leave, both being Pure -- and neither is
// spilled: a fill is not a compute, so a fill feeding a generic is not a
// schedule boundary. Both are cloned into each group instead, `tensor.empty`
// included, so the second group's fill does not read the first group's empty.
//
// Every store and every load gets its own construct_access_tile, and each memory
// op its own memory view and tile-id arithmetic: each group is extracted into a
// schedule of its own, so an operation two groups read belongs to neither.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set_tile = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_whole = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

// One buffer for the exp result and nothing else: a spilled fill would show up
// here as a second entry.
// CHECK: module attributes {ktdp.hbm_roundtrip_buffers = [{element_type = f32, shape = array<i64: 12, 64, 64>}]}
// CHECK-LABEL: func.func @two_computes(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index, %[[SPILL:.*]]: index) attributes

// Group 0 reads the input, fills its own accumulator and writes the spill buffer.
// CHECK:         %[[LOADED:.*]] = ktdp.load
// CHECK:         %[[EMPTY0:.*]] = tensor.empty()
// CHECK:         %[[FILL0:.*]] = linalg.fill {{.*}} outs(%[[EMPTY0]]
// CHECK:         %[[EXP:.*]] = linalg.generic {{.*}} outs(%[[FILL0]]
// CHECK:           spyreop.exp
// CHECK:         %[[SPILL_VIEW_W:.*]] = ktdp.construct_memory_view %[[SPILL]], sizes: [12, 64, 64], strides: [4096, 64, 1]
// CHECK:         %[[SPILL_TILE_W:.*]] = ktdp.construct_access_tile %[[SPILL_VIEW_W]]
// CHECK:         ktdp.store %[[EXP]], %[[SPILL_TILE_W]]

// Group 1 reads it back through a view and access tile of its own, and has a
// fill and an empty of its own too.
// CHECK:         %[[SPILL_VIEW_R:.*]] = ktdp.construct_memory_view %[[SPILL]], sizes: [12, 64, 64], strides: [4096, 64, 1]
// CHECK:         %[[SPILL_TILE_R:.*]] = ktdp.construct_access_tile %[[SPILL_VIEW_R]]
// CHECK:         %[[BACK:.*]] = ktdp.load %[[SPILL_TILE_R]]
// CHECK:         %[[EMPTY1:.*]] = tensor.empty()
// CHECK:         %[[FILL1:.*]] = linalg.fill {{.*}} outs(%[[EMPTY1]]
// CHECK:         %[[SQRT:.*]] = linalg.generic {{.*}} ins(%[[BACK]]{{.*}} outs(%[[FILL1]]
// CHECK:           spyreop.sqrt
// CHECK:         ktdp.store %[[SQRT]]
module {
  func.func @two_computes(%base_in: index, %base_out: index) attributes {grid = [2]} {
    %zero = arith.constant 0 : index
    %init_value = arith.constant 0.000000e+00 : f32
    %tid = ktdp.get_compute_tile_id : index

    %view_in = ktdp.construct_memory_view %base_in, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set_whole, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%tid * 6, %zero, %zero] {access_tile_order = #map, access_tile_set = #set_tile} : memref<12x64x64xf32> -> !ktdp.access_tile<6x64x64xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set_whole, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%tid * 6, %zero, %zero] {access_tile_order = #map, access_tile_set = #set_tile} : memref<12x64x64xf32> -> !ktdp.access_tile<6x64x64xindex>

    %in = ktdp.load %tile_in : <6x64x64xindex> -> tensor<6x64x64xf32>
    // One empty and one fill for both computes, as CSE leaves them.
    %init = tensor.empty() : tensor<6x64x64xf32>
    %filled = linalg.fill ins(%init_value : f32) outs(%init : tensor<6x64x64xf32>) -> tensor<6x64x64xf32>
    %mid = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%in : tensor<6x64x64xf32>) outs(%filled : tensor<6x64x64xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<6x64x64xf32>
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mid : tensor<6x64x64xf32>) outs(%filled : tensor<6x64x64xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<6x64x64xf32>
    ktdp.store %result, %tile_out : tensor<6x64x64xf32>, <6x64x64xindex>
    return
  }
}

// -----

// Three chained computes: two edges, so two buffers. One buffer per edge, never
// shared -- the middle compute's own input buffer is not handed on to its output
// even though nothing reads it again, because there is no liveness reasoning
// here at all.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

// Two buffers, so two entries in the list and two added arguments. The closing
// paren on the signature is what makes that a count rather than a lower bound.
// CHECK: module attributes {ktdp.hbm_roundtrip_buffers = [{element_type = f32, shape = array<i64: 128>}, {element_type = f32, shape = array<i64: 128>}]}
// CHECK-LABEL: func.func @three_computes(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index, %[[SPILL0:.*]]: index, %[[SPILL1:.*]]: index) attributes
module {
  func.func @three_computes(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>

    %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf32>
    %init0 = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%in : tensor<128xf32>) outs(%init0 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    %init1 = tensor.empty() : tensor<128xf32>
    %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%init1 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<128xf32>
    %init2 = tensor.empty() : tensor<128xf32>
    %c = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%b : tensor<128xf32>) outs(%init2 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    ktdp.store %c, %tile_out : tensor<128xf32>, <128xindex>
    return
  }
}

// -----

// A DAG rather than a chain, which is where three things happen that a chain
// never produces:
//
//   1. `%a` is read by both later computes, so its buffer takes one store and
//      *two* loads, in two different compute groups. In a chain every buffer has
//      exactly one reader.
//   2. the last compute reads `%a` and `%b`, produced two computes back and one
//      back, so one group performs two independent reloads.
//   3. the kernel's own input load is read by the first and last computes, so it
//      is cloned per group, address cone and all -- the load-cloning branch of
//      privatizeComputeInputs, which a single-use input load never reaches.
//
// The arithmetic is irrelevant; the operand graph is the point.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

// Two edges (`%a` and `%b`), so two buffers and two added arguments, however many
// loads each buffer ends up with.
// CHECK: module attributes {ktdp.hbm_roundtrip_buffers = [{element_type = f32, shape = array<i64: 128>}, {element_type = f32, shape = array<i64: 128>}]}
// CHECK-LABEL: func.func @dag_not_a_chain(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index, %[[SPILL_A:.*]]: index, %[[SPILL_B:.*]]: index) attributes

// Group 0: reads the input through a load of its own and writes %a's buffer.
// CHECK:         %[[IN_VIEW0:.*]] = ktdp.construct_memory_view %[[IN]],
// CHECK:         %[[IN_TILE0:.*]] = ktdp.construct_access_tile %[[IN_VIEW0]]
// CHECK:         %[[X0:.*]] = ktdp.load %[[IN_TILE0]]
// CHECK:         %[[A:.*]] = linalg.generic {{.*}} ins(%[[X0]]
// CHECK:           spyreop.exp
// CHECK:         %[[A_VIEW_W:.*]] = ktdp.construct_memory_view %[[SPILL_A]],
// CHECK:         %[[A_TILE_W:.*]] = ktdp.construct_access_tile %[[A_VIEW_W]]
// CHECK:         ktdp.store %[[A]], %[[A_TILE_W]]

// Group 1: the first of %a's two readers, then writes %b's buffer.
// CHECK:         %[[A_VIEW_R1:.*]] = ktdp.construct_memory_view %[[SPILL_A]],
// CHECK:         %[[A_TILE_R1:.*]] = ktdp.construct_access_tile %[[A_VIEW_R1]]
// CHECK:         %[[A_BACK1:.*]] = ktdp.load %[[A_TILE_R1]]
// CHECK:         %[[B:.*]] = linalg.generic {{.*}} ins(%[[A_BACK1]]
// CHECK:           spyreop.sqrt
// CHECK:         %[[B_VIEW_W:.*]] = ktdp.construct_memory_view %[[SPILL_B]],
// CHECK:         %[[B_TILE_W:.*]] = ktdp.construct_access_tile %[[B_VIEW_W]]
// CHECK:         ktdp.store %[[B]], %[[B_TILE_W]]

// Group 2: its own clone of the input load, %a's *second* load, and %b's -- three
// loads off three separate cones, feeding one generic.
// CHECK:         %[[IN_VIEW2:.*]] = ktdp.construct_memory_view %[[IN]],
// CHECK:         %[[IN_TILE2:.*]] = ktdp.construct_access_tile %[[IN_VIEW2]]
// CHECK:         %[[X2:.*]] = ktdp.load %[[IN_TILE2]]
// CHECK:         %[[A_VIEW_R2:.*]] = ktdp.construct_memory_view %[[SPILL_A]],
// CHECK:         %[[A_TILE_R2:.*]] = ktdp.construct_access_tile %[[A_VIEW_R2]]
// CHECK:         %[[A_BACK2:.*]] = ktdp.load %[[A_TILE_R2]]
// CHECK:         %[[B_VIEW_R:.*]] = ktdp.construct_memory_view %[[SPILL_B]],
// CHECK:         %[[B_TILE_R:.*]] = ktdp.construct_access_tile %[[B_VIEW_R]]
// CHECK:         %[[B_BACK:.*]] = ktdp.load %[[B_TILE_R]]
// CHECK:         %[[C:.*]] = linalg.generic {{.*}} ins(%[[X2]], %[[A_BACK2]], %[[B_BACK]]
// CHECK:         ktdp.store %[[C]],
module {
  func.func @dag_not_a_chain(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>

    // One load of the input, read by the first and the last compute.
    %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf32>
    %init0 = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%in : tensor<128xf32>) outs(%init0 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    %init1 = tensor.empty() : tensor<128xf32>
    %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%init1 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<128xf32>
    %init2 = tensor.empty() : tensor<128xf32>
    %c = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%in, %a, %b : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%init2 : tensor<128xf32>) {
    ^bb0(%x0: f32, %x1: f32, %x2: f32, %out: f32):
      %m = arith.mulf %x0, %x1 : f32
      %s = arith.addf %m, %x2 : f32
      linalg.yield %s : f32
    } -> tensor<128xf32>
    ktdp.store %c, %tile_out : tensor<128xf32>, <128xindex>
    return
  }
}

// -----

// An edge on a grid of rank greater than one is reported, not silently skipped:
// the roundtrip is needed and this pass cannot address a spill slab with more
// than one compute-tile index. Reported here rather than left to fail later
// without naming the limitation.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>

module {
  // expected-error @below {{HbmRoundtrip: this kernel hands a value from one compute to another, which has to be routed through HBM, but this pass is a temporary stand-in that handles linalg.generic, linalg.fill and linalg.reduce on a rank-1 grid only: its grid has rank 2}}
  func.func @rank_2_grid_is_reported(%base_in: index, %base_out: index) attributes {grid = [2, 3]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero, %zero] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero, %zero] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>

    %in = ktdp.load %tile_in : <4x128xindex> -> tensor<4x128xf32>
    %init0 = tensor.empty() : tensor<4x128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%in : tensor<4x128xf32>) outs(%init0 : tensor<4x128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<4x128xf32>
    %init1 = tensor.empty() : tensor<4x128xf32>
    %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x128xf32>) outs(%init1 : tensor<4x128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<4x128xf32>
    ktdp.store %b, %tile_out : tensor<4x128xf32>, <4x128xindex>
    return
  }
}

// -----

// A single compute has no compute-to-compute edge, so the pass leaves the
// function exactly as it found it -- same arguments, no module attribute, and no
// extra memory operations.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

// CHECK-NOT: ktdp.hbm_roundtrip_buffers
// CHECK-LABEL: func.func @one_compute(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index) attributes
// CHECK-COUNT-1: ktdp.load
// CHECK-COUNT-1: ktdp.store
// CHECK-NOT:     ktdp.load
module {
  func.func @one_compute(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>

    %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf32>
    %init = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%in : tensor<128xf32>) outs(%init : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    ktdp.store %a, %tile_out : tensor<128xf32>, <128xindex>
    return
  }
}

// -----

// `linalg.reduce` is in scope, so a reduce handing its result to a generic gets
// an ordinary roundtrip: one buffer for the reduced tensor, a store after the
// reduce and a load before the generic. The scheduler's named-op allowlist for
// compute is `linalg.{add,mul,sub,max,min,reduce,generic,yield}` and `reduce` is
// in it, so there is nothing special about it here. Note the buffer is the
// *reduced* shape, 4xf32, not the input's.
#whole = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0) -> (d0)>
#set_2d = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set_1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>

// CHECK: module attributes {ktdp.hbm_roundtrip_buffers = [{element_type = f32, shape = array<i64: 4>}]}
// CHECK-LABEL: func.func @reduce_feeding_a_generic(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index, %[[SPILL:.*]]: index) attributes
// CHECK:         %[[SUM:.*]] = linalg.reduce
// CHECK:         %[[VIEW_W:.*]] = ktdp.construct_memory_view %[[SPILL]],
// CHECK:         %[[TILE_W:.*]] = ktdp.construct_access_tile %[[VIEW_W]]
// CHECK:         ktdp.store %[[SUM]], %[[TILE_W]]
// CHECK:         %[[VIEW_R:.*]] = ktdp.construct_memory_view %[[SPILL]],
// CHECK:         %[[TILE_R:.*]] = ktdp.construct_access_tile %[[VIEW_R]]
// CHECK:         %[[BACK:.*]] = ktdp.load %[[TILE_R]]
// CHECK:         linalg.generic {{.*}} ins(%[[BACK]]
module {
  func.func @reduce_feeding_a_generic(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set_2d, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero, %zero] {access_tile_order = #whole, access_tile_set = #set_2d} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [4], strides: [1] {coordinate_set = #set_1d, memory_space = #ktdp.memory_space<global>} : memref<4xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #row, access_tile_set = #set_1d} : memref<4xf32> -> !ktdp.access_tile<4xindex>

    %in = ktdp.load %tile_in : <4x128xindex> -> tensor<4x128xf32>
    %init0 = tensor.empty() : tensor<4xf32>
    %sum = linalg.reduce ins(%in : tensor<4x128xf32>) outs(%init0 : tensor<4xf32>) dimensions = [1]
      (%x: f32, %acc: f32) {
        %s = arith.addf %x, %acc : f32
        linalg.yield %s : f32
      }
    %init1 = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {indexing_maps = [#row, #row], iterator_types = ["parallel"]} ins(%sum : tensor<4xf32>) outs(%init1 : tensor<4xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.sqrt %x : f32
      linalg.yield %e : f32
    } -> tensor<4xf32>
    ktdp.store %result, %tile_out : tensor<4xf32>, <4xindex>
    return
  }
}

// -----

// `linalg.broadcast` is what stays out of scope, and an edge through one is
// reported. It is the named op that matters in practice: LowerComputeOps emits it
// for `tt.broadcast` / `tt.expand_dims`, and the scheduler's named-op allowlist
// does not contain it -- nor `linalg.transpose` or `linalg.matmul`.
#whole = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0) -> (d0)>
#set_2d = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set_1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>

module {
  // expected-error @below {{HbmRoundtrip: this kernel hands a value from one compute to another, which has to be routed through HBM, but this pass is a temporary stand-in that handles linalg.generic, linalg.fill and linalg.reduce on a rank-1 grid only: it contains linalg.broadcast}}
  func.func @broadcast_is_reported(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [4], strides: [1] {coordinate_set = #set_1d, memory_space = #ktdp.memory_space<global>} : memref<4xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #row, access_tile_set = #set_1d} : memref<4xf32> -> !ktdp.access_tile<4xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set_2d, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero, %zero] {access_tile_order = #whole, access_tile_set = #set_2d} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>

    %in = ktdp.load %tile_in : <4xindex> -> tensor<4xf32>
    %init0 = tensor.empty() : tensor<4xf32>
    %a = linalg.generic {indexing_maps = [#row, #row], iterator_types = ["parallel"]} ins(%in : tensor<4xf32>) outs(%init0 : tensor<4xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.sqrt %x : f32
      linalg.yield %e : f32
    } -> tensor<4xf32>
    %init1 = tensor.empty() : tensor<4x128xf32>
    %wide = linalg.broadcast ins(%a : tensor<4xf32>) outs(%init1 : tensor<4x128xf32>) dimensions = [1]
    ktdp.store %wide, %tile_out : tensor<4x128xf32>, <4x128xindex>
    return
  }
}

// -----

// An out-of-scope op with no edge to break is left alone and *not* reported --
// the same `linalg.broadcast`, with nothing handing one compute's result to
// another. This is the case that pins the order: scope is only asked about once
// there is work to do, so a kernel that compiles today does not start failing
// merely for containing an op this pass cannot place.
#whole = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0) -> (d0)>
#set_2d = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set_1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>

// CHECK-NOT: ktdp.hbm_roundtrip_buffers
// CHECK-LABEL: func.func @broadcast_without_an_edge(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index) attributes
// CHECK-COUNT-1: ktdp.load
// CHECK:         linalg.broadcast
// CHECK-COUNT-1: ktdp.store
// CHECK-NOT:     ktdp.load
module {
  func.func @broadcast_without_an_edge(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [4], strides: [1] {coordinate_set = #set_1d, memory_space = #ktdp.memory_space<global>} : memref<4xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #row, access_tile_set = #set_1d} : memref<4xf32> -> !ktdp.access_tile<4xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set_2d, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero, %zero] {access_tile_order = #whole, access_tile_set = #set_2d} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>

    %in = ktdp.load %tile_in : <4xindex> -> tensor<4xf32>
    %init = tensor.empty() : tensor<4x128xf32>
    %wide = linalg.broadcast ins(%in : tensor<4xf32>) outs(%init : tensor<4x128xf32>) dimensions = [1]
    ktdp.store %wide, %tile_out : tensor<4x128xf32>, <4x128xindex>
    return
  }
}
