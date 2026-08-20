// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout | FileCheck %s

// Regression test: retypeChain must COMPOSE permutations when the forward
// walk from a physicalized ktdp.load crosses TWO linalg.transpose ops (with
// an elementwise op between them so canonicalize doesn't fold them first),
// not overwrite the recorded permutation with just the second one.
//
// A is stick-on-M; %a chains tt.trans[1,0] -> arith.negf -> tt.trans[1,0],
// composing to the identity. dispatchSource must still synthesize the same
// layout-driven [1, 0] transpose as the transpose-free case (see
// matmul_both_parallel in rewrite-descriptor-layout-matmul.mlir) — an
// uncomposed overwrite would instead corrupt dimRoles and change the
// synthesized transpose/matmul shape.
//
// CHECK lines regenerated with utils/generate-test-checks.py. Key assertions
// if hand-editing:
//   - both A_LOAD and B_LOAD are physicalized to tensor<1x64x64xf32>
//   - exactly ONE linalg.transpose (permutation = [1, 0]) is synthesized,
//     applied to the slice extracted from A's (negf'd) physical load
//   - the transposed A operand and the untransposed B operand feed into
//     exactly one linalg.matmul
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   tt.func @chained_transpose_matmul(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf32>
// CHECK:           %[[VAL_6:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_9:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_10:.*]] = arith.divsi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_12:.*]] = arith.remsi %[[VAL_7]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%[[VAL_10]], %[[VAL_8]], %[[VAL_12]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<1x64x64xf32> -> !ktdp.access_tile<1x64x64xindex>
// CHECK:           %[[A_LOAD:.*]] = ktdp.load %[[VAL_13]] : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[VAL_15:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[A_NEG:.*]] = arith.negf %[[A_LOAD]] : tensor<1x64x64xf32>
// CHECK:           %[[VAL_17:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[VAL_18:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_19:.*]] = ktdp.construct_memory_view %[[VAL_18]], sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf32>
// CHECK:           %[[VAL_20:.*]] = ktdp.construct_memory_view %[[VAL_18]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
// CHECK:           %[[VAL_21:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_22:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_23:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_24:.*]] = arith.divsi %[[VAL_22]], %[[VAL_23]] : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_26:.*]] = arith.remsi %[[VAL_22]], %[[VAL_25]] : index
// CHECK:           %[[VAL_27:.*]] = ktdp.construct_access_tile %[[VAL_19]]{{\[}}%[[VAL_24]], %[[VAL_21]], %[[VAL_26]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<1x64x64xf32> -> !ktdp.access_tile<1x64x64xindex>
// CHECK:           %[[B_LOAD:.*]] = ktdp.load %[[VAL_27]] : <1x64x64xindex> -> tensor<1x64x64xf32>
// CHECK:           %[[VAL_29:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_30:.*]] = ktdp.construct_memory_view %[[VAL_29]], sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
// CHECK:           %[[VAL_31:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_32:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_33:.*]] = ktdp.construct_access_tile %[[VAL_30]]{{\[}}%[[VAL_31]], %[[VAL_32]]] {access_tile_order = #[[$ATTR_1]], access_tile_set = #[[$ATTR_3]]} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
// CHECK:           %[[C_LOAD:.*]] = ktdp.load %[[VAL_33]] : <64x64xindex> -> tensor<64x64xf32>
// CHECK:           %[[VAL_35:.*]] = arith.constant 0 : index
// The A operand slice is extracted from the negf'd physical load (still at
// physical, unswapped shape)...
// CHECK:           %[[A_SLICE:.*]] = tensor.extract_slice %[[A_NEG]][0, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<1x64x64xf32> to tensor<64x64xf32>
// ...and dispatchSource synthesizes exactly ONE net transpose [1, 0] for A
// (the layout-driven one), because the two user-level tt.trans ops on the
// source chain compose to identity and must cancel out, NOT compose with the
// layout transpose into something else.
// CHECK:           %[[A_EMPTY:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[A_T:.*]] = linalg.transpose ins(%[[A_SLICE]] : tensor<64x64xf32>) outs(%[[A_EMPTY]] : tensor<64x64xf32>) permutation = [1, 0]
// CHECK:           %[[B_SLICE:.*]] = tensor.extract_slice %[[B_LOAD]][0, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<1x64x64xf32> to tensor<64x64xf32>
// CHECK:           %[[MM:.*]] = linalg.matmul ins(%[[A_T]], %[[B_SLICE]] : tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[C_LOAD]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK:           %[[VAL_41:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_42:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_43:.*]] = ktdp.construct_access_tile %[[VAL_30]]{{\[}}%[[VAL_41]], %[[VAL_42]]] {access_tile_order = #[[$ATTR_1]], access_tile_set = #[[$ATTR_3]]} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
// CHECK:           ktdp.store %[[MM]], %[[VAL_43]] : tensor<64x64xf32>, <64x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @chained_transpose_matmul(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[M=64, K=64] (square, so a role swap stays shape-compatible) stick-on-M(64):
  // phys [M/64, K, M%64] = [1, 64, 64]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  // Chained transposes with an elementwise op between them, so canonicalize
  // cannot fold them into a single transpose (or into a no-op) before
  // --rewrite-descriptor-layout runs. Net permutation = identity.
  %a_t1 = tt.trans %a {order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
  %a_neg = arith.negf %a_t1 : tensor<64x64xf32>
  %a_t2 = tt.trans %a_neg {order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>

  // B[K=64, N=64] stick-on-N(64): phys [N/64, K, N%64] = [1, 64, 64]
  %b_desc = tt.make_tensor_descriptor %b_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %b_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  // C unannotated [64x64]
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  %d = tt.dot %a_t2, %b, %c : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>

  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<64x64xf32>, tensor<64x64xf32>
  tt.return
}
