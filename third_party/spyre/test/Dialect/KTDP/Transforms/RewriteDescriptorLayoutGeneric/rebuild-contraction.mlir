// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s
// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s --check-prefix=NOLOOP

// rebuild-reduction.mlir's rule, with a second input: the iterator kinds follow
// the dims, so splitting the reduced dim gives two reduction LOOP DIMS -- dims of
// the generic's iteration space, which is what "loop" means throughout this
// directory. Never an scf.for, and the difference is the point below.
//
// What that buys here is worth stating on its own, because a contraction is where
// one would expect a loop over sticks. There is none: this pass emits no scf.for
// and no tensor.extract_slice whatever the contracted dim's stick count, since a
// linalg reduction over two loop dims IS that loop, expressed where the scheduler
// can still see the whole thing as one op. The NOLOOP prefix carries that negative
// half of each case; the named-op pass emits a stick loop, a rescaling multiply
// and a pair of slices for these same kernels.
//
// Eleven cases, all positive. Cases 1 to 4 vary where the contracted dim sits:
// at one stick, at one stick inside a batched kernel that has a program-id loop
// of its own, at two sticks, and an accumulator carried across an enclosing
// scf.for as its iter_arg.
//
// Cases 5 and 6 physicalize the contraction's OWN RESULT -- the output view is
// annotated and no generic mediates the store, so the `outs` and the result type
// change and the result map names the output's stick pair directly.
//
// Cases 7 to 9 vary WHICH dim each operand splits rather than how far: a parallel
// dim on both, the same reduced dim on both, and one of each. Case 10 chains two
// contractions through a scratchpad intermediate on no layout, and case 11 splits
// TWO dims of a single operand, taking it to rank 5.
//
// What this pass DECLINES is not shaped like a contraction -- a named
// linalg.matmul is in invalid-layout.mlir and the malformed-input declines are in
// invalid-ktir.mlir -- so no case here is expected to fail.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which emits only positive CHECKs and would drop
// every NOLOOP line, and cannot segment case 2 at all because of its nested
// scf.for.

// CHECK: #[[$S1_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$S1_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$S1_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: #[[$S1_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0 * 64 + d2, d4)>
// CHECK: #[[$S1_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3 * 64 + d4)>
// CHECK: #[[$S1_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$S1_SET2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// Case 1 -- a rank-2 contraction, both split extents equal to 1.
//
// A 64-wide dim at stick 64 is exactly one stick, so both splits have extent 1.
// The rule does not inspect extents, so the split is emitted anyway and this
// kernel's output is structurally identical to the two-stick case below. The
// cost is the composite ks * 64 + kl on B where a plain ks would do, accepted
// deliberately: an extent test is exactly the special case that would stop this
// being one rule.
//
// The output view carries no layout, so the result stays logical by design
// and its map names only the unsplit dims, composing the split ones.
//
// Input produced from the @matmul_single_stick kernel in
// RewriteDescriptorLayout/matmul.mlir, by lowering it to pre-pass IR and running
// --linalg-morph-ops=named-to-generic. That named fixture's physical types are
// the oracle for the physicalization half; the compute half differs on purpose,
// since this pass emits no stick loop and no slicing.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// The loop and the slicing the named path emits for this kernel are gone: the
// generic's own maps name the stick dims, so there is nothing left to iterate or
// slice.
// NOLOOP-LABEL:   tt.func @matmul_single_stick(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_single_stick(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$S1_SET3]], memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_3]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_3]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%[[VAL_7]], %[[VAL_3]], %[[VAL_9]]] {access_tile_order = #[[$S1_ID3]], access_tile_set = #[[$S1_SET3]]} : memref<1x64x64xf32> -> !ktdp.access_tile<1x64x64xindex>
// CHECK:           %[[VAL_11:.*]] = ktdp.load %[[VAL_10]] : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_12]], sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$S1_SET3]], memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_15:.*]] = arith.divsi %[[VAL_3]], %[[VAL_14]] : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_17:.*]] = arith.remsi %[[VAL_3]], %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = ktdp.construct_access_tile %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_3]], %[[VAL_17]]] {access_tile_order = #[[$S1_ID3]], access_tile_set = #[[$S1_SET3]]} : memref<1x64x64xf32> -> !ktdp.access_tile<1x64x64xindex>
// CHECK:           %[[VAL_19:.*]] = ktdp.load %[[VAL_18]] : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[VAL_20:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_21:.*]] = ktdp.construct_memory_view %[[VAL_20]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$S1_SET2]], memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
// CHECK:           %[[VAL_22:.*]] = ktdp.construct_access_tile %[[VAL_21]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {access_tile_order = #[[$S1_ID2]], access_tile_set = #[[$S1_SET2]]} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
// CHECK:           %[[VAL_23:.*]] = ktdp.load %[[VAL_22]] : <64x64xindex> -> tensor<64x64xf32>
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#[[$S1_A]], #[[$S1_B]], #[[$S1_C]]], iterator_types = ["reduction", "parallel", "reduction", "parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_19]] : tensor<1x64x64xf32>, tensor<1x64x64xf32>) outs(%[[VAL_23]] : tensor<64x64xf32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[VAL_28:.*]] = arith.mulf %[[VAL_25]], %[[VAL_26]] : f32
// CHECK:             %[[VAL_29:.*]] = arith.addf %[[VAL_27]], %[[VAL_28]] : f32
// CHECK:             linalg.yield %[[VAL_29]] : f32
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           %[[VAL_30:.*]] = ktdp.construct_access_tile %[[VAL_21]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {access_tile_order = #[[$S1_ID2]], access_tile_set = #[[$S1_SET2]]} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
// CHECK:           ktdp.store %[[VAL_24]], %[[VAL_30]] : tensor<64x64xf32>, <64x64xindex>
// CHECK:           tt.return
// CHECK:         }
  tt.func @matmul_single_stick(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %4 = ktdp.load %3 : <64x64xindex> -> tensor<64x64xf32>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %9 = ktdp.load %8 : <64x64xindex> -> tensor<64x64xf32>
    %10 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %11 = ktdp.construct_memory_view %10, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %12 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %13 = ktdp.load %12 : <64x64xindex> -> tensor<64x64xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %9 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %16 = arith.mulf %in, %in_0 : f32
      %17 = arith.addf %out, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<64x64xf32>
    %15 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %14, %15 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// CHECK: #[[$S2_ID4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$S2_A:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK: #[[$S2_B:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d1, d0 * 64 + d3, d5)>
// CHECK: #[[$S2_C:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4 * 64 + d5)>
// CHECK: #[[$S2_TRUNC_IN:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0 * 64 + d3)>
// CHECK: #[[$S2_SET_A:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$S2_SET_B:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$S2_SET_C:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>

// Case 2 -- a batch contraction inside a kernel that already has a loop of its
// own.
//
// K=128 at stick 64, so A splits K into two real sticks while B holds it whole at
// 128; N=64 gives one stick on B and on the output. A therefore names the two K
// loops (d4, d5) as a plain projected permutation and B carries the composite,
// while the accumulator -- an unmarked splat constant, so it stays logical --
// composes the N split it does not carry. Two reduction loop dims for the split K;
// batch, M and the N split are parallel.
//
// This is the case that proves the loop elimination, because the kernel's own
// program-id scf.for is the only loop in the output: the pass neither adds one for
// the K sticks nor rescales the one that was already there. For this same kernel
// the named path emits an scf.for over the K sticks plus an arith.muli and a pair
// of extract_slices, and none of that appears here.
//
// The truncf between the contraction and the store is spelled as a generic, which
// is what convert_elementwise_to_linalg produces; as a bare arith.truncf on a
// tensor this pass declines it by name, since it rewrites only generics. It is
// also what carries the result to the store's shape: the contraction's outs stays
// logical, and this generic re-sticks it. At the store the data tile then agrees
// with the access tile, with no widening stage -- see rebuild-composite.mlir.
//
// Input produced from RewriteDescriptorLayout/batch-matmul.mlir by lowering it to
// pre-pass IR. The checks are hand-written: generate-test-checks.py cannot
// segment a kernel with a nested scf.for.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func public @bmm_matmul_kernel(
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           scf.for
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func public @bmm_matmul_kernel(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f16>, %[[VAL_1:.*]]: !tt.ptr<f16>, %[[VAL_2:.*]]: !tt.ptr<f16>) attributes {noinline = false} {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : tensor<4x64x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_8:.*]] = tt.get_num_programs x : i32
// CHECK:           %[[VAL_9:.*]] = arith.divsi %[[VAL_8]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_7]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_12:.*]] = arith.minsi %[[VAL_11]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f16> to index
// CHECK:           %[[VAL_14:.*]] = ktdp.construct_memory_view %[[VAL_13]], sizes: [2, 4, 64, 64], strides: [16384, 4096, 64, 1] {coordinate_set = #[[$S2_SET_A]], memory_space = #ktdp.memory_space<global>} : memref<2x4x64x64xf16>
// CHECK:           %[[VAL_15:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f16> to index
// CHECK:           %[[VAL_16:.*]] = ktdp.construct_memory_view %[[VAL_15]], sizes: [1, 4, 128, 64], strides: [32768, 8192, 64, 1] {coordinate_set = #[[$S2_SET_B]], memory_space = #ktdp.memory_space<global>} : memref<1x4x128x64xf16>
// CHECK:           %[[VAL_17:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f16> to index
// CHECK:           %[[VAL_18:.*]] = ktdp.construct_memory_view %[[VAL_17]], sizes: [1, 4, 64, 64], strides: [16384, 4096, 64, 1] {coordinate_set = #[[$S2_SET_C]], memory_space = #ktdp.memory_space<global>} : memref<1x4x64x64xf16>
// CHECK:           scf.for %[[VAL_19:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_6]]  : i32 {
// CHECK:             %[[VAL_20:.*]] = arith.muli %[[VAL_19]], %[[VAL_5]] : i32
// CHECK:             %[[VAL_21:.*]] = arith.index_cast %[[VAL_20]] : i32 to index
// CHECK:             %[[VAL_22:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_23:.*]] = arith.divsi %[[VAL_3]], %[[VAL_22]] : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_25:.*]] = arith.remsi %[[VAL_3]], %[[VAL_24]] : index
// CHECK:             %[[VAL_26:.*]] = ktdp.construct_access_tile %[[VAL_14]]{{\[}}%[[VAL_23]], %[[VAL_21]], %[[VAL_3]], %[[VAL_25]]] {access_tile_order = #[[$S2_ID4]], access_tile_set = #[[$S2_SET_A]]} : memref<2x4x64x64xf16> -> !ktdp.access_tile<2x4x64x64xindex>
// CHECK:             %[[VAL_27:.*]] = ktdp.load %[[VAL_26]] : <2x4x64x64xindex> -> tensor<2x4x64x64xf16>
// CHECK:             %[[VAL_28:.*]] = arith.index_cast %[[VAL_20]] : i32 to index
// CHECK:             %[[VAL_29:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_30:.*]] = arith.divsi %[[VAL_3]], %[[VAL_29]] : index
// CHECK:             %[[VAL_31:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_32:.*]] = arith.remsi %[[VAL_3]], %[[VAL_31]] : index
// CHECK:             %[[VAL_33:.*]] = ktdp.construct_access_tile %[[VAL_16]]{{\[}}%[[VAL_30]], %[[VAL_28]], %[[VAL_3]], %[[VAL_32]]] {access_tile_order = #[[$S2_ID4]], access_tile_set = #[[$S2_SET_B]]} : memref<1x4x128x64xf16> -> !ktdp.access_tile<1x4x128x64xindex>
// CHECK:             %[[VAL_34:.*]] = ktdp.load %[[VAL_33]] : <1x4x128x64xindex> -> tensor<1x4x128x64xf16>
// CHECK:             %[[VAL_35:.*]] = linalg.generic {indexing_maps = [#[[$S2_A]], #[[$S2_B]], #[[$S2_C]]], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} ins(%[[VAL_27]], %[[VAL_34]] : tensor<2x4x64x64xf16>, tensor<1x4x128x64xf16>) outs(%[[VAL_4]] : tensor<4x64x64xf32>) {
// CHECK:             ^bb0(%[[VAL_36:.*]]: f16, %[[VAL_37:.*]]: f16, %[[VAL_38:.*]]: f32):
// CHECK:               %[[VAL_39:.*]] = arith.extf %[[VAL_36]] : f16 to f32
// CHECK:               %[[VAL_40:.*]] = arith.extf %[[VAL_37]] : f16 to f32
// CHECK:               %[[VAL_41:.*]] = arith.mulf %[[VAL_39]], %[[VAL_40]] : f32
// CHECK:               %[[VAL_42:.*]] = arith.addf %[[VAL_38]], %[[VAL_41]] : f32
// CHECK:               linalg.yield %[[VAL_42]] : f32
// CHECK:             } -> tensor<4x64x64xf32>
// CHECK:             %[[VAL_43:.*]] = tensor.empty() : tensor<1x4x64x64xf16>
// CHECK:             %[[VAL_44:.*]] = linalg.generic {indexing_maps = [#[[$S2_TRUNC_IN]], #[[$S2_ID4]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_35]] : tensor<4x64x64xf32>) outs(%[[VAL_43]] : tensor<1x4x64x64xf16>) {
// CHECK:             ^bb0(%[[VAL_45:.*]]: f32, %[[VAL_46:.*]]: f16):
// CHECK:               %[[VAL_47:.*]] = arith.truncf %[[VAL_45]] : f32 to f16
// CHECK:               linalg.yield %[[VAL_47]] : f16
// CHECK:             } -> tensor<1x4x64x64xf16>
// CHECK:             %[[VAL_48:.*]] = arith.index_cast %[[VAL_20]] : i32 to index
// CHECK:             %[[VAL_49:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_50:.*]] = arith.divsi %[[VAL_3]], %[[VAL_49]] : index
// CHECK:             %[[VAL_51:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_52:.*]] = arith.remsi %[[VAL_3]], %[[VAL_51]] : index
// CHECK:             %[[VAL_53:.*]] = ktdp.construct_access_tile %[[VAL_18]]{{\[}}%[[VAL_50]], %[[VAL_48]], %[[VAL_3]], %[[VAL_52]]] {access_tile_order = #[[$S2_ID4]], access_tile_set = #[[$S2_SET_C]]} : memref<1x4x64x64xf16> -> !ktdp.access_tile<1x4x64x64xindex>
// CHECK:             ktdp.store %[[VAL_44]], %[[VAL_53]] : tensor<1x4x64x64xf16>, <1x4x64x64xindex>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @bmm_matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64x64xf32>
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.divsi %1, %1 : i32
    %3 = arith.muli %0, %2 : i32
    %4 = arith.addi %3, %2 : i32
    %5 = arith.minsi %4, %c1_i32 : i32
    %6 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
    %7 = ktdp.construct_memory_view %6, sizes: [4, 64, 128], strides: [8192, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 2, 0, 1, 2>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>}} : memref<4x64x128xf16>
    %9 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
    %10 = ktdp.construct_memory_view %9, sizes: [4, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 2, 0, 1, 2>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>}} : memref<4x128x64xf16>
    %12 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
    %13 = ktdp.construct_memory_view %12, sizes: [4, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 2, 0, 1, 2>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>}} : memref<4x64x64xf16>
    scf.for %arg3 = %3 to %5 step %c1_i32  : i32 {
      %15 = arith.muli %arg3, %c4_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = ktdp.construct_access_tile %7[%16, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x64x128xf16> -> !ktdp.access_tile<4x64x128xindex>
      %18 = ktdp.load %17 : <4x64x128xindex> -> tensor<4x64x128xf16>
      %19 = arith.index_cast %15 : i32 to index
      %20 = ktdp.construct_access_tile %10[%19, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<4x128x64xf16> -> !ktdp.access_tile<4x128x64xindex>
      %21 = ktdp.load %20 : <4x128x64xindex> -> tensor<4x128x64xf16>
      %22 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%18, %21 : tensor<4x64x128xf16>, tensor<4x128x64xf16>) outs(%cst : tensor<4x64x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %out: f32):
        %26 = arith.extf %in : f16 to f32
        %27 = arith.extf %in_0 : f16 to f32
        %28 = arith.mulf %26, %27 : f32
        %29 = arith.addf %out, %28 : f32
        linalg.yield %29 : f32
      } -> tensor<4x64x64xf32>
      %e = tensor.empty() : tensor<4x64x64xf16>
      %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<4x64x64xf32>) outs(%e : tensor<4x64x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %t = arith.truncf %in : f32 to f16
        linalg.yield %t : f16
      } -> tensor<4x64x64xf16>
      %24 = arith.index_cast %15 : i32 to index
      %25 = ktdp.construct_access_tile %13[%24, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
      ktdp.store %23, %25 : tensor<4x64x64xf16>, <4x64x64xindex>
    }
    tt.return
  }
}

// -----

// Case 3 -- a rank-2 contraction whose contracted dim is more than one stick.
//
// K=128 at stick 64 gives two sticks on A, and B holds K whole at 128. This is
// the case one might expect to need a loop over the K sticks, and it does not:
// the split gives the rebuilt domain a SECOND reduction loop, A names both
// halves as a plain projected permutation, and B -- which holds K whole --
// carries the composite. A linalg reduction over two loop dims is the loop, so
// the pass emits no scf.for here any more than in the degenerate case above.

// CHECK: #[[$S3_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$S3_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$S3_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: #[[$S3_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0 * 64 + d2, d4)>
// CHECK: #[[$S3_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3 * 64 + d4)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_two_sticks(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
//
// CHECK-LABEL:   tt.func @matmul_two_sticks(
// A[M=64, K=128] stick-on-K(64) -> two real sticks.
// CHECK:           ktdp.construct_memory_view %{{.*}}, sizes: [2, 64, 64], strides: [4096, 64, 1]
// CHECK-SAME:        : memref<2x64x64xf32>
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// B[K=128, N=64] stick-on-N(64) -> one stick on N, K held whole at 128.
// CHECK:           ktdp.construct_memory_view %{{.*}}, sizes: [1, 128, 64], strides: [8192, 64, 1]
// CHECK-SAME:        : memref<1x128x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <1x128x64xindex> -> tensor<1x128x64xf32>
// The accumulator carries no layout, so it stays logical and composes the N
// split it does not carry.
// CHECK:           %[[CL:.*]] = ktdp.load %{{.*}} : <64x64xindex> -> tensor<64x64xf32>
// Two reduction loop dims for the split K; M and the N split are parallel.
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S3_A]], #[[$S3_B]], #[[$S3_C]]], iterator_types = ["reduction", "parallel", "reduction", "parallel", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<2x64x64xf32>, tensor<1x128x64xf32>) outs(%[[CL]] : tensor<64x64xf32>) {
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<64x64xf32>, <64x64xindex>
  tt.func @matmul_two_sticks(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %9 = ktdp.load %8 : <128x64xindex> -> tensor<128x64xf32>
    %10 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %11 = ktdp.construct_memory_view %10, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %12 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %13 = ktdp.load %12 : <64x64xindex> -> tensor<64x64xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %9 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%13 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %16 = arith.mulf %in, %in_0 : f32
      %17 = arith.addf %out, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<64x64xf32>
    %15 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %14, %15 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 4 -- the accumulator is an enclosing scf.for's iter_arg.
//
// The attention P @ V inner loop: B is stick-on-N and the accumulator is not
// annotated, so it holds N whole and its map carries the composite d2 * 64 + d3.
// The accumulator's type therefore does not change, and the loop's iter_arg, its
// result type and the scf.yield operand are all left as they are -- the rank change
// lives entirely inside the generic's maps. The loop itself is untouched, as in
// subscripts-direct.mlir case 2.
//
// An accumulator that DID have to be retyped is another matter: it is a block
// argument, with no producer to restate, and block-arg-outs.mlir has that.

// CHECK: #[[$S4_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$S4_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$S4_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK: #[[$S4_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
// CHECK: #[[$S4_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 64 + d3)>
// CHECK: #[[$S4_SET_A:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$S4_SET_B:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$S4_SET_C:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>

#a_m = affine_map<(d0, d1, d2) -> (d0, d2)>
#b_m = affine_map<(d0, d1, d2) -> (d2, d1)>
#c_m = affine_map<(d0, d1, d2) -> (d0, d1)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#sa = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// The loop this pass did not touch: one scf.for, its iter_arg and result still at
// the logical type, and no slicing anywhere.
// NOLOOP-LABEL:   tt.func @matmul_loop_carried_acc(
// NOLOOP:           scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<64x128xf32>)
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_loop_carried_acc(
// CHECK:           %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
// A is unannotated and stays [64, 64]; B is stick-on-N and becomes [2, 64, 64].
// CHECK:           %[[AV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$S4_SET_A]], memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
// CHECK:           %[[BV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$S4_SET_B]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[OV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 128], strides: [128, 1] {coordinate_set = #[[$S4_SET_C]], memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
// CHECK:           %[[RES:.*]] = scf.for %{{.*}} iter_args(%[[ACC:.*]] = %[[CST]]) -> (tensor<64x128xf32>) {
// CHECK:             %[[AL:.*]] = ktdp.load %{{.*}} : <64x64xindex> -> tensor<64x64xf32>
// CHECK:             %[[BL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:             %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S4_A]], #[[$S4_B]], #[[$S4_C]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<64x64xf32>, tensor<2x64x64xf32>) outs(%[[ACC]] : tensor<64x128xf32>) {
// CHECK:             } -> tensor<64x128xf32>
// CHECK:             scf.yield %[[R]] : tensor<64x128xf32>
// CHECK:           }
// CHECK:           ktdp.store %[[RES]], %{{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK:           tt.return
tt.func @matmul_loop_carried_acc(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 64], strides: [64, 1] {coordinate_set = #sa, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f32> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sb, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sb, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %res = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %cst) -> (tensor<64x128xf32>) {
    %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sa} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %al = ktdp.load %at : <64x64xindex> -> tensor<64x64xf32>
    %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sb} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf32>
    %r = linalg.generic {indexing_maps = [#a_m, #b_m, #c_m], iterator_types = ["parallel", "parallel", "reduction"]} ins(%al, %bl : tensor<64x64xf32>, tensor<64x128xf32>) outs(%acc : tensor<64x128xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %p = arith.mulf %x, %y : f32
      %s = arith.addf %z, %p : f32
      linalg.yield %s : f32
    } -> tensor<64x128xf32>
    scf.yield %r : tensor<64x128xf32>
  }
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sb} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %res, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 5 -- the contraction's OWN RESULT is physicalized.
//
// The output view is annotated and nothing stands between the contraction and the
// store, so the contraction's `outs` IS the physicalized load: its type becomes
// [N/64, M, N%64] and so does the result's. That is the whole difference from
// cases 1 and 3, whose output views carry no layout, and from case 2, where a
// truncf generic mediates and absorbs the shape change instead.
//
// A[M=64, K=128] stick-on-K(64) -> [2, 64, 64]; B[K=128, N=64] stick-on-N(64) ->
// [1, 128, 64]; D[M=64, N=64] stick-on-N(64) -> [1, 64, 64]. Five loop dims: K's
// stick and lane are the reductions, N's stick and lane and M are parallel. A and
// the result name their own pairs plainly; only B, holding K whole, carries a
// composite.

// CHECK: #[[$S5_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$S5_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>
// CHECK: #[[$S5_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0 * 64 + d3, d4)>
// CHECK: #[[$S5_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_annotated_output(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP-NOT:       tensor.insert_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_annotated_output(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <1x128x64xindex> -> tensor<1x128x64xf32>
// The accumulator is annotated, so it is loaded at the physical shape.
// CHECK:           %[[DL:.*]] = ktdp.load %{{.*}} : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S5_A]], #[[$S5_B]], #[[$S5_C]]], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<2x64x64xf32>, tensor<1x128x64xf32>) outs(%[[DL]] : tensor<1x64x64xf32>) {
// CHECK:           } -> tensor<1x64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<1x64x64xf32>, <1x64x64xindex>
// CHECK:           tt.return
  tt.func @matmul_annotated_output(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 6 -- the same physicalized result, with both inputs split on K.
//
// B is stick-on-K here rather than stick-on-N, so it splits the reduced dim the
// way A does and names the K pair plainly; what it holds whole is N, and that is
// the composite its map carries. The result still names N's own pair. The loop
// dims and iterator kinds are identical to case 5 -- which dim an operand holds
// whole moves the arithmetic between maps and changes nothing else.
//
// A[M=64, K=128] stick-on-K(64) -> [2, 64, 64]; B[K=128, N=64] stick-on-K(64) ->
// [2, 64, 64]; D[M=64, N=64] stick-on-N(64) -> [1, 64, 64].

// CHECK: #[[$S6_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>
// CHECK: #[[$S6_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d4, d3)>
// CHECK: #[[$S6_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_k_split_annotated_output(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP-NOT:       tensor.insert_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_k_split_annotated_output(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[DL:.*]] = ktdp.load %{{.*}} : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S6_A]], #[[$S6_B]], #[[$S6_C]]], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%[[DL]] : tensor<1x64x64xf32>) {
// CHECK:           } -> tensor<1x64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<1x64x64xf32>, <1x64x64xindex>
// CHECK:           tt.return
  tt.func @matmul_k_split_annotated_output(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 7 -- both operands split a PARALLEL dim, and the result carries both
// composites.
//
// A is stick-on-M and B stick-on-N, so neither splits the reduced K: K stays one
// loop dim and the two splits add a parallel pair each. The unannotated result
// holds both M and N whole, so its single map carries TWO composites -- the only
// case in this directory where one map does.
//
// A[M=64, K=128] stick-on-M(64) -> [1, 128, 64]; B[K=128, N=64] stick-on-N(64) ->
// [1, 128, 64]; C[64, 64] on no layout.

// CHECK: #[[$S7_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: #[[$S7_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d1, d4)>
// CHECK: #[[$S7_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d2, d3 * 64 + d4)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_both_parallel(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_both_parallel(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <1x128x64xindex> -> tensor<1x128x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <1x128x64xindex> -> tensor<1x128x64xf32>
// CHECK:           %[[CL:.*]] = ktdp.load %{{.*}} : <64x64xindex> -> tensor<64x64xf32>
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S7_A]], #[[$S7_B]], #[[$S7_C]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<1x128x64xf32>, tensor<1x128x64xf32>) outs(%[[CL]] : tensor<64x64xf32>) {
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<64x64xf32>, <64x64xindex>
// CHECK:           tt.return
  tt.func @matmul_both_parallel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 8 -- both operands split the SAME reduced dim, so no map carries
// arithmetic at all.
//
// A and B are both stick-on-K at the same width, so K's stick and lane are the
// only dims either splits and both name the pair plainly. The unannotated result
// mentions neither, having no K in its map to begin with. Four loop dims, two of
// them reductions, and not one composite: this is rebuild-passthrough.mlir's
// "nothing to add" at a contraction.
//
// A[M=64, K=128] stick-on-K(64) -> [2, 64, 64]; B[K=128, N=64] stick-on-K(64) ->
// [2, 64, 64]; C[64, 64] on no layout.

// CHECK: #[[$S8_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$S8_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$S8_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_both_k_split(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_both_k_split(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[CL:.*]] = ktdp.load %{{.*}} : <64x64xindex> -> tensor<64x64xf32>
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S8_A]], #[[$S8_B]], #[[$S8_C]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%[[CL]] : tensor<64x64xf32>) {
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<64x64xf32>, <64x64xindex>
// CHECK:           tt.return
  tt.func @matmul_both_k_split(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 9 -- one operand splits a parallel dim, the other the reduced dim.
//
// A is stick-on-M and B stick-on-K, so the two splits land on dims of different
// KINDS and each operand holds whole exactly what the other split: A's map
// carries K's composite, B's map names its own K pair, and the unannotated result
// carries M's composite. Five loop dims, iterators alternating around the two
// pairs.
//
// A[M=64, K=128] stick-on-M(64) -> [1, 128, 64]; B[K=128, N=64] stick-on-K(64) ->
// [2, 64, 64]; C[64, 64] on no layout.

// CHECK: #[[$S9_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d4, d2)>
// CHECK: #[[$S9_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK: #[[$S9_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d2, d3)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_a_parallel_b_ksplit(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_a_parallel_b_ksplit(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <1x128x64xindex> -> tensor<1x128x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[CL:.*]] = ktdp.load %{{.*}} : <64x64xindex> -> tensor<64x64xf32>
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S9_A]], #[[$S9_B]], #[[$S9_C]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[AL]], %[[BL]] : tensor<1x128x64xf32>, tensor<2x64x64xf32>) outs(%[[CL]] : tensor<64x64xf32>) {
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<64x64xf32>, <64x64xindex>
// CHECK:           tt.return
  tt.func @matmul_a_parallel_b_ksplit(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 10 -- two contractions chained through a scratchpad intermediate.
//
// D = A @ (B @ C), where the inner result is an ordinary tensor on no layout that
// the outer contraction reads directly. Each generic is rebuilt against its OWN
// operands' layouts and nothing is carried between them: the scratchpad keeps its
// logical [64, 64] and appears in the second generic's map as a composite, exactly
// as any unannotated operand would. The two rebuilt domains differ in rank, which
// is what shows there is no shared state -- five loop dims for the inner pair,
// four for the outer.
//
// A, B and C all stick-on-dim-1(64) at one stick each -> [1, 64, 64]; the output
// view is on no layout, so the store stays logical.

// CHECK: #[[$S10_IN_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: #[[$S10_IN_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0 * 64 + d2, d4)>
// CHECK: #[[$S10_IN_C:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3 * 64 + d4)>
// CHECK: #[[$S10_OUT_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$S10_OUT_S:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d2, d3)>
// CHECK: #[[$S10_OUT_D:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#mA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mC = affine_map<(d0, d1, d2) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @matmul_chain(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @matmul_chain(
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[CL:.*]] = ktdp.load %{{.*}} : <1x64x64xindex> -> tensor<1x64x64xf32>
// The scratchpad's init is on no layout and stays logical, so the inner result
// does too.
// CHECK:           %[[BC:.*]] = linalg.generic {indexing_maps = [#[[$S10_IN_A]], #[[$S10_IN_B]], #[[$S10_IN_C]]], iterator_types = ["reduction", "parallel", "reduction", "parallel", "parallel"]} ins(%[[BL]], %[[CL]] : tensor<1x64x64xf32>, tensor<1x64x64xf32>) outs(%{{.*}} : tensor<64x64xf32>) {
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <1x64x64xindex> -> tensor<1x64x64xf32>
// The outer generic reads the scratchpad at its logical shape, so its map carries
// the composite -- and its domain is a different rank from the inner one's.
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S10_OUT_A]], #[[$S10_OUT_S]], #[[$S10_OUT_D]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%[[AL]], %[[BC]] : tensor<1x64x64xf32>, tensor<64x64xf32>) outs(%{{.*}} : tensor<64x64xf32>) {
// CHECK:           } -> tensor<64x64xf32>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<64x64xf32>, <64x64xindex>
// CHECK:           tt.return
  tt.func @matmul_chain(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %3 = ktdp.load %2 : <64x64xindex> -> tensor<64x64xf32>
    %4 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %7 = ktdp.load %6 : <64x64xindex> -> tensor<64x64xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %8 = linalg.generic {indexing_maps = [#mA, #mB, #mC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%cst : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %p = arith.mulf %x, %y : f32
      %s = arith.addf %z, %p : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    %9 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %10 = ktdp.construct_memory_view %9, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %11 = ktdp.construct_access_tile %10[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %12 = ktdp.load %11 : <64x64xindex> -> tensor<64x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %13 = linalg.generic {indexing_maps = [#mA, #mB, #mC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12, %8 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%cst_0 : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %p = arith.mulf %x, %y : f32
      %s = arith.addf %z, %p : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    %14 = builtin.unrealized_conversion_cast %arg3 : !tt.ptr<f32> to index
    %15 = ktdp.construct_memory_view %14, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %16 = ktdp.construct_access_tile %15[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %13, %16 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 11 -- TWO independent splits on ONE operand, taking it to rank 5.
//
// A batch contraction whose A splits both M and K at 64, so A alone contributes
// four of the rebuilt domain's six loop dims and lands at [M/64, K/64, B, M%64,
// K%64] = [2, 2, 2, 64, 64]. B splits only K, so it names K's pair and holds N
// whole; the unannotated result holds M whole and carries M's composite. Nothing
// in the rule counts splits per operand, which is what this case says: two splits
// on one operand are two splits, threaded independently, and the physical dim
// order is the layout's rather than the logical one grown on the right -- A's map
// is a non-monotone projected permutation for that reason.
//
// A[B=2, M=128, K=128] split on M(64) and K(64) -> [2, 2, 2, 64, 64];
// B[B=2, K=128, N=64] split on K(64) -> [2, 2, 64, 64];
// C[2, 128, 64] on no layout. Iterators [p, p, r, p, r, p]: batch, M's stick, K's
// stick, M's lane, K's lane, N.

// CHECK: #[[$S11_A:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d0, d3, d4)>
// CHECK: #[[$S11_B:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d4, d5)>
// CHECK: #[[$S11_C:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 64 + d3, d5)>

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#bA = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bB = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#bC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
// NOLOOP-LABEL:   tt.func @parallel_floor_rank5(
// NOLOOP-NOT:       scf.for
// NOLOOP-NOT:       tensor.extract_slice
// NOLOOP-NOT:       tensor.insert_slice
// NOLOOP:           tt.return
// CHECK-LABEL:   tt.func @parallel_floor_rank5(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x2x2x64x64xindex> -> tensor<2x2x2x64x64xf16>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <2x2x64x64xindex> -> tensor<2x2x64x64xf16>
// CHECK:           %[[CL:.*]] = ktdp.load %{{.*}} : <2x128x64xindex> -> tensor<2x128x64xf16>
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$S11_A]], #[[$S11_B]], #[[$S11_C]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%[[AL]], %[[BL]] : tensor<2x2x2x64x64xf16>, tensor<2x2x64x64xf16>) outs(%[[CL]] : tensor<2x128x64xf16>) {
// CHECK:           } -> tensor<2x128x64xf16>
// CHECK:           ktdp.store %[[R]], %{{.*}} : tensor<2x128x64xf16>, <2x128x64xindex>
// CHECK:           tt.return
  tt.func @parallel_floor_rank5(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
    %1 = ktdp.construct_memory_view %0, sizes: [2, 128, 128], strides: [16384, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 2, 0, 1, 2>, phys_op = array<i64: 1, 1, 0, 2, 2>, phys_arg = array<i64: 64, 64, 0, 64, 64>}} : memref<2x128x128xf16>
    %2 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x128x128xf16> -> !ktdp.access_tile<2x128x128xindex>
    %3 = ktdp.load %2 : <2x128x128xindex> -> tensor<2x128x128xf16>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
    %5 = ktdp.construct_memory_view %4, sizes: [2, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1, 2>, phys_op = array<i64: 1, 0, 2, 0>, phys_arg = array<i64: 64, 0, 64, 0>}} : memref<2x128x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x128x64xf16> -> !ktdp.access_tile<2x128x64xindex>
    %7 = ktdp.load %6 : <2x128x64xindex> -> tensor<2x128x64xf16>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
    %9 = ktdp.construct_memory_view %8, sizes: [2, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x128x64xf16>
    %10 = ktdp.construct_access_tile %9[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x128x64xf16> -> !ktdp.access_tile<2x128x64xindex>
    %11 = ktdp.load %10 : <2x128x64xindex> -> tensor<2x128x64xf16>
    %12 = linalg.generic {indexing_maps = [#bA, #bB, #bC], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<2x128x128xf16>, tensor<2x128x64xf16>) outs(%11 : tensor<2x128x64xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %p = arith.mulf %x, %y : f16
      %s = arith.addf %z, %p : f16
      linalg.yield %s : f16
    } -> tensor<2x128x64xf16>
    %13 = ktdp.construct_access_tile %9[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x128x64xf16> -> !ktdp.access_tile<2x128x64xindex>
    ktdp.store %12, %13 : tensor<2x128x64xf16>, <2x128x64xindex>
    tt.return
  }
}
