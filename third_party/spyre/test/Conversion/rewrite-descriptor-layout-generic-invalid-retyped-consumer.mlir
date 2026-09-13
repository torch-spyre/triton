// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -verify-diagnostics

// A value the rewrite retyped, read by something that is not a linalg.generic.
//
// The generic's result is retyped because the store it feeds requires it, and it
// also escapes through tt.return. Without this check the pass would leave a
// return whose operand no longer matches the function's result type, and the only
// complaint would come from a verifier naming neither this pass nor the op.
//
// The check is on the consumer, not on the value: by then the value IS at
// physical rank, so asking about the value would answer yes and let it through.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @two_consumers(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) -> tensor<64x128xf32> {
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
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  // expected-error @below {{rewrite-descriptor-layout-generic: this op reads a value the rewrite retyped, but the rewrite restates only linalg.generic; spell this op as one}}
  tt.return %r : tensor<64x128xf32>
}
}
