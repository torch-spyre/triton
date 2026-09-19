// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -verify-diagnostics

// A decline must name the op, not surface as a verifier failure.
//
// LowerComputeOps lowers tt.dot to a NAMED linalg.matmul, which is fixed at
// logical rank and so cannot carry a stick dim. This pass rewrites only generics,
// so it rejects the op up front, before Phase 1 retypes the load underneath it.
// The ordering is the point: retyping first would leave a rank mismatch that
// MLIR's own verifier reports against an indexing map, naming neither this pass
// nor what it could not restate.
//
// A[M=128, K=64] stick-on-M(64) -> physical [M/64, K, M%64] = [2, 64, 64].

#id = affine_map<(d0, d1) -> (d0, d1)>
#sa = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @named_matmul_declines(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  %av = ktdp.construct_memory_view %ai, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sa, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16>
  %ad = builtin.unrealized_conversion_cast %av : memref<128x64xf16> to !tt.tensordesc<128x64xf16>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <128x64xf16>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #sa} : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
  %al = ktdp.load %at : <128x64xindex> -> tensor<128x64xf16>

  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f16> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 64], strides: [64, 1] {coordinate_set = #sb, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #sb} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
  %bl = ktdp.load %bt : <64x64xindex> -> tensor<64x64xf16>

  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f16> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sa, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #sa} : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
  %cl = ktdp.load %ct : <128x64xindex> -> tensor<128x64xf16>

  // expected-error @below {{rewrite-descriptor-layout-generic: this op reads a value on a physicalized chain, but the rewrite restates only linalg.generic; spell this op as one}}
  %d = linalg.matmul ins(%al, %bl : tensor<128x64xf16>, tensor<64x64xf16>) outs(%cl : tensor<128x64xf16>) -> tensor<128x64xf16>

  %st = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #sa} : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
  ktdp.store %d, %st : tensor<128x64xf16>, <128x64xindex>
  tt.return
}
}
