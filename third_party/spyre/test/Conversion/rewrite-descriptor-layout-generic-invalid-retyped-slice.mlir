// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -verify-diagnostics

// A value the rewrite retyped, read by something that is not a linalg.generic.
//
// tensor.extract_slice is the reachable shape of this: LowerComputeOps lowers
// tt.split to a pair of them, so the op arrives from the same pipeline that
// produces the rest of this IR. It names its offsets and sizes per logical dim,
// which is exactly what physicalizing invalidates -- the operand becomes
// <2x64x64> while the op still says <64x128>. Without this check the only
// complaint would come from a verifier naming neither this pass nor the op.
//
// The consumer here reads the GENERIC's result, not the load's. That is what
// makes it land in checkAllConsistent rather than the up-front consumer scan,
// which only looks at what a ktdp.load feeds.
//
// The check is on the consumer, not on the value: by then the value IS at
// physical rank, so asking about the value would answer yes and let it through.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#s2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @sliced_consumer(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %odd = builtin.unrealized_conversion_cast %ov : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %odd {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  // The store is what makes %r physicalize in the first place; the slice below
  // is the second reader, and the one that cannot follow.
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  // expected-error @below {{rewrite-descriptor-layout-generic: this op reads a value the rewrite retyped, but the rewrite restates only linalg.generic, so this op still names the logical type}}
  %half = tensor.extract_slice %r[0, 0] [64, 64] [1, 1] : tensor<64x128xf32> to tensor<64x64xf32>
  %ht = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x64xindex>
  ktdp.store %half, %ht : tensor<64x64xf32>, <64x64xindex>
  tt.return
}
}
