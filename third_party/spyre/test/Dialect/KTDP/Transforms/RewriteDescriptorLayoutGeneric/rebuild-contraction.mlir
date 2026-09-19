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
// Three cases, all positive: the contracted dim at one stick, at one stick inside
// a batched kernel that has a program-id loop of its own, and at two sticks. What
// this pass DECLINES is not shaped like a contraction -- a named linalg.matmul is
// in invalid-layout.mlir and the malformed-input declines are in invalid-ktir.mlir
// -- so no case here is expected to fail.
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
