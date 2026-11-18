// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --materialize-pinned-buffers | FileCheck %s --check-prefix=INHERIT
// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PHYS

// A pin's buffer is a memory view like any other and leaves `spyrecode` PHYSICAL.
// What is particular about it is only that the author cannot annotate it -- it is
// the compiler's buffer, so there is nothing for them to state a layout on -- and
// the answer is that the compiler decides it from the pin's neighbours, the way a
// `linalg.generic` has its domain decided.
//
// Two runs, because the mechanism and its effect are separate claims. The first
// stops after the pin pass and shows the layout COPIED onto the `ct_local` view,
// still logical; the second runs the whole stage and shows that view stick-tiled by
// the layout pass, which needed no change to reach it.
//
// Both kernels here are from the #211 review. The second is the one that did not
// compile: a pinned value stored through an annotated descriptor had no generic
// between it and the store for the layout pass to restate, and the diagnostic it got
// said to annotate the source -- which is the pin's buffer, and not annotatable.
//
// `M = 64, N = 128` at fp16, stick-on-N, so a physical extent is
// [ceil(N/64), M, 64] = [2, 64, 64] and the buffer holds the same 8192 elements its
// logical shape does. A non-multiple would pad, which is why `checkOffset` measures
// the physical count once a buffer carries a layout.

// The pinned value feeds a COMPUTE before its store, which is the case that compiled
// before this and still does. It compiles for a reason worth keeping: the sqrt
// generic has a layout on its result and none on its operand, and the rewrite reads
// one per operand, so a logical operand inside a physicalized domain was read at
// `stick * width + lane`. What changes is that there is no longer a logical operand
// to read that way -- the buffer is physical, and both ends agree.
// The one line is deliberate: every view in this kernel is a construct_memory_view
// under the same layout, so the LX one is named by its memory space rather than by
// its position among them.
// INHERIT-LABEL: func.func @pin_then_compute(
// INHERIT:       ktdp.construct_memory_view {{.*}}sizes: [64, 128], strides: [128, 1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// INHERIT-SAME:    tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}
// INHERIT-NOT:   tts.pin
//
// PHYS-LABEL:    func.func @pin_then_compute(
// PHYS:          ktdp.construct_memory_view {{.*}}sizes: [2, 64, 64], strides: [4096, 64, 1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// PHYS-NOT:      tts.tensor_layout
// PHYS-NOT:      tts.pin

module {
  tt.func public @pin_then_compute(%in_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f16>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %m = arith.constant 64 : i32
    %n = arith.constant 128 : i32
    %sm = arith.constant 128 : i64
    %sn = arith.constant 1 : i64
    %in_desc = tt.make_tensor_descriptor %in_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %in_desc {phys_src = array<i64: 1, 0, 1>,
                                phys_op = array<i64: 1, 0, 2>,
                                phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>,
                                 phys_op = array<i64: 1, 0, 2>,
                                 phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %x = tt.descriptor_load %in_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
    %e = math.exp %x : tensor<64x128xf16>
    tts.pin %e {memory_space = "ct_local", offset = 0 : i32} : tensor<64x128xf16>
    %y = math.sqrt %e : tensor<64x128xf16>
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %y : !tt.tensordesc<64x128xf16>, tensor<64x128xf16>
    tt.return
  }
}

// -----
// The same value STORED OUT as well as consumed, which is the shape the review asked
// about and the one that used to fail. Keeping a pinned value for later use while
// also writing it out is the case that matters; a store immediately after a pin and
// nothing else would be a round trip with no purpose.
//
// The store needs no generic now, because the buffer it reads from and the descriptor
// it writes to are physical under the same layout -- so the reordering that had no
// vehicle is not a reordering any more.
// The one line is deliberate: every view in this kernel is a construct_memory_view
// under the same layout, so the LX one is named by its memory space rather than by
// its position among them.
// INHERIT-LABEL: func.func @pin_store_and_compute(
// INHERIT:       ktdp.construct_memory_view {{.*}}sizes: [64, 128], strides: [128, 1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// INHERIT-SAME:    tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}
// INHERIT-NOT:   tts.pin
//
// PHYS-LABEL:    func.func @pin_store_and_compute(
// PHYS:          ktdp.construct_memory_view {{.*}}sizes: [2, 64, 64], strides: [4096, 64, 1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// PHYS-NOT:      tts.tensor_layout
// PHYS-NOT:      tts.pin

module {
  tt.func public @pin_store_and_compute(%in_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f16>, %out2_ptr: !tt.ptr<f16>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %m = arith.constant 64 : i32
    %n = arith.constant 128 : i32
    %sm = arith.constant 128 : i64
    %sn = arith.constant 1 : i64
    %in_desc = tt.make_tensor_descriptor %in_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %in_desc {phys_src = array<i64: 1, 0, 1>,
                                phys_op = array<i64: 1, 0, 2>,
                                phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>,
                                 phys_op = array<i64: 1, 0, 2>,
                                 phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %out2_desc = tt.make_tensor_descriptor %out2_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %out2_desc {phys_src = array<i64: 1, 0, 1>,
                                  phys_op = array<i64: 1, 0, 2>,
                                  phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %x = tt.descriptor_load %in_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
    %e = math.exp %x : tensor<64x128xf16>
    tts.pin %e {memory_space = "ct_local", offset = 0 : i32} : tensor<64x128xf16>
    tt.descriptor_store %out2_desc[%c0_i32, %c0_i32], %e : !tt.tensordesc<64x128xf16>, tensor<64x128xf16>
    %y = math.sqrt %e : tensor<64x128xf16>
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %y : !tt.tensordesc<64x128xf16>, tensor<64x128xf16>
    tt.return
  }
}

// -----
// The FORWARD direction, carrying the one layout a backward walk CANNOT produce.
//
// The reduction here is the same as `@layout_not_carried_off_a_split` in the pass's
// own directory, and it reduces ON the stick axis. The split dim goes and the stick
// structure with it, so the result is re-stuck by a SPLAT -- `phys_op
// [identity, splat]` over `phys_src [0, 0]`, replicating the surviving dim across a
// stick's 64 lanes (RewriteDescriptorLayoutGeneric/rebuild-reduction.mlir, case 2).
// That is exactly why the backward walk stops at such a reduce: nothing upstream
// states the width to broadcast over, so a carry would have to invent one.
//
// Forward it is not invented but READ. The store states the splat layout, the buffer
// takes it verbatim, and the physical form is [4, 64] -- which is what makes the two
// cases a pair rather than two tests: the same layout is refused off the pin's
// producing side and inherited off its consuming side.
//
// The store is the pinned value's only use, deliberately: what is under test is the
// DIRECTION a layout arrives from, and a second consumer would add a second
// neighbour to reason about. It does mean the value makes a round trip through the
// buffer, which is what puts the physical view in the output to check.
//
// Hand-written KTIR rather than a TTIR kernel, unlike the two above, because nothing
// in tree drives a reduction through `--spyre-ttir-to-ktir` and the subject is the
// layout rather than how a reduce reaches KTIR. Verified that the stage passes KTIR
// input through unchanged, so both RUN lines above still apply to it.
// INHERIT-LABEL: func.func @layout_carried_from_a_restuck_store(
// INHERIT:       ktdp.construct_memory_view {{.*}}sizes: [4], strides: [1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// INHERIT-SAME:    tts.tensor_layout = {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>}
// INHERIT-NOT:   tts.pin
//
// PHYS-LABEL:    func.func @layout_carried_from_a_restuck_store(
// PHYS:          ktdp.construct_memory_view {{.*}}sizes: [4, 64], strides: [64, 1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// PHYS-NOT:      tts.tensor_layout
// PHYS-NOT:      tts.pin

#sin = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#sout = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>
module {
  tt.func @layout_carried_from_a_restuck_store(%src: index, %dst: index) {
    %c0 = arith.constant 0 : index
    %v = ktdp.construct_memory_view %src, sizes: [4, 128], strides: [128, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<4x128xf16>
    %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<4x128xf16> -> !ktdp.access_tile<4x128xindex>
    %x = ktdp.load %t : <4x128xindex> -> tensor<4x128xf16>
    %init = tensor.empty() : tensor<4xf16>
    %r = linalg.reduce { arith.addf } ins(%x : tensor<4x128xf16>) outs(%init : tensor<4xf16>) dimensions = [1]
        {tts.pin = {memory_space = "ct_local", offset = 0 : i32}}
    // The re-stuck destination: rank-1 logical, physical [4, 64] by the splat.
    %o = ktdp.construct_memory_view %dst, sizes: [4], strides: [1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_src = array<i64: 0, 0>, phys_op = array<i64: 0, 3>, phys_arg = array<i64: 0, 64>}} : memref<4xf16>
    %ot = ktdp.construct_access_tile %o[%c0] {access_tile_order = #id1, access_tile_set = #sout} : memref<4xf16> -> !ktdp.access_tile<4xindex>
    ktdp.store %r, %ot : tensor<4xf16>, <4xindex>
    tt.return
  }
}
