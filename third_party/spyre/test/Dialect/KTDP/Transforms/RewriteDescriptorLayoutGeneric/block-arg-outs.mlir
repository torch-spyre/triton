// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file > %t 2>&1 && echo TOOL_SUCCEEDED >> %t || true
// RUN: FileCheck %s --allow-empty < %t

// An outs operand that is a BLOCK ARGUMENT, which the rewrite declines without
// saying anything.
//
// retypeToPhysical restates the producer of each operand it has to retype:
// tensor.empty, a splat constant, linalg.fill, or another linalg.generic. A block
// argument has no producer, and the final branch there is
// `v.getDefiningOp() ? emitError(...) : failure()` -- so for a value with no
// defining op the pass returns failure with no diagnostic attached. The tool then
// exits non-zero having printed nothing, which is what the two RUN lines above pin
// between them: the marker is appended only on a zero exit, and the output holds no
// diagnostic either way.
//
// This file records current behaviour. It is not a claim that declining is right,
// only that declining SILENTLY is what happens today, and that a future diagnostic
// will have to change this file.
//
// CHECK-NOT: TOOL_SUCCEEDED
// CHECK-NOT: error

// Case 1 -- the outs is an scf.for iter_arg.
//
// A contraction accumulating across the loop, the annotated destination forcing the
// result to physical rank 3, so the accumulator has to be retyped with it. The
// result is read only by the store, which is what keeps the up-front consumer check
// quiet and lets the retype be reached; `scf.yield %acc` keeps the loop well formed
// without giving the result a second reader.

#a_m = affine_map<(d0, d1, d2) -> (d0, d2)>
#b_m = affine_map<(d0, d1, d2) -> (d2, d1)>
#c_m = affine_map<(d0, d1, d2) -> (d0, d1)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#sa = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @outs_is_iter_arg(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
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
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sb, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
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
    %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sb} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
    scf.yield %acc : tensor<64x128xf32>
  }
  tt.return
}
}

// -----

// Case 2 -- the outs is a tt.func argument.
//
// The same branch reached without a loop: an accumulator handed in as a tensor
// argument. Both ends are annotated and the generic is elementwise, so nothing but
// the outs's own producer is in question -- and there is none.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @outs_is_func_arg(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>, %acc: tensor<64x128xf32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%acc : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.addf %x, %y : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}
