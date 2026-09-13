// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout-generic -verify-diagnostics

// An annotated gather is declined, and the decline names what is unsupported.
//
// A scope reduction from RewriteDescriptorLayout, which does physicalize an
// indirect access tile (for a rank-2 gather, with the gather dim required to be
// unsplit). Nothing about the map rebuild rules that out; the tile's own
// per-dim subscript maps simply have to be restated at physical rank too, and no
// fixture exercises that here yet. Declining by name is the honest state.

module {
tt.func @gather_with_layout(%data_ptr: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c32_i64 = arith.constant 32 : i64

  // Index descriptor: 32-element 1-D tensor holding row indices.
  %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%c32_i32], [%c1_i64]
      : !tt.ptr<i32>, !tt.tensordesc<32xi32>
  %x_offsets = tt.descriptor_load %idx_desc[%c0_i32] : !tt.tensordesc<32xi32> -> tensor<32xi32>

  // Data descriptor: [512, 128] with stick-on-N layout (stick=64).
  //   phys_src=[1, 0, 1] phys_op=[1, 0, 2] phys_arg=[64, 0, 64]
  //   => physical shape [N/64, M, N%64] = [2, 512, 64]
  %data_desc = tt.make_tensor_descriptor %data_ptr, [%c512_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<1x128xf32>
  tt.spyre_tensor_layout %data_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<1x128xf32>

  // Gather 32 non-contiguous rows.
  // expected-error @below {{spyre_tensor_layout: physicalizing an indirect access tile is not supported by rewrite-descriptor-layout-generic}}
  %gathered = tt.descriptor_gather %data_desc[%x_offsets, %c0_i32]
      : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>

  // Store the gathered result to a physical-annotated output.
  %out_desc = tt.make_tensor_descriptor %out_ptr, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<32x128xf32>
  tt.spyre_tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x128xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %gathered : !tt.tensordesc<32x128xf32>, tensor<32x128xf32>

  tt.return
}
}
