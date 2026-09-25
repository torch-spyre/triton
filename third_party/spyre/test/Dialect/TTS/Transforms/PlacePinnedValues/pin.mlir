// RUN: spyre-triton-opt %s -split-input-file --place-pinned-values | FileCheck %s

// A `tts.pin` becoming the buffer it asked for.
//
// Four things happen to a pinned value, and all four are the point:
//
//   1. a ktdp.construct_memory_view is built over the value's own shape, in the
//      pin's memory space, at the pinned address;
//   2. the value is STORED to it, where the pin stood;
//   3. each remaining use reads it back through its own access tile and
//      ktdp.load -- so the consumer no longer names the produced value at all;
//   4. the marker is erased.
//
// (3) is the one worth asserting negatively, and the cases below do: a pass that
// built the buffer and left the consumers reading the register would satisfy
// every positive CHECK here and place nothing. The guard is that the consumer's
// operand is the load's result, which is what `math.sqrt %[[LOADED]]` pins.
//
// One view is shared by the store and every load, with a fresh tile per access.
// That is what LowerDescriptorMemory does for a descriptor, and the shape known
// to reach the device; see the pass for the alternative it declines.
//
// Input is hand-written post-LowerDescriptorMemory IR with one pass in the RUN
// line, so what the pass under test receives is in the file. Note there are no
// rules of dashes anywhere here: -split-input-file matches its marker as a
// substring, so a line of dashes would start a new chunk.

// A constant address: one offset, the same on every core. 4096 elements at fp16
// is byte 8192, which is stick-aligned, and the buffer is the value's shape with
// row-major strides -- not the producer's layout, since a pin states none.
// CHECK-LABEL: tt.func @constant_address
// CHECK: %[[E:.*]] = math.exp
// CHECK: %[[OFF:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [4, 64], strides: [64, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<ct_local>} : memref<4x64xf16>
// CHECK: %[[STILE:.*]] = ktdp.construct_access_tile %[[VIEW]]
// CHECK: ktdp.store %[[E]], %[[STILE]]
// CHECK: %[[LTILE:.*]] = ktdp.construct_access_tile %[[VIEW]]
// CHECK: %[[LOADED:.*]] = ktdp.load %[[LTILE]]
// CHECK: math.sqrt %[[LOADED]]
// CHECK-NOT: tts.pin
tt.func @constant_address(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// An address affine in the program id. The expression is NOT folded to a number:
// the view's offset is an SSA `index` operand, so the arithmetic flows into it
// and only the width changes. What the pass computes from `(base, stride)` is the
// range it checks, not the address it builds.
// CHECK-LABEL: tt.func @affine_address
// CHECK: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK: %[[MUL:.*]] = arith.muli %[[PID]]
// CHECK: %[[ADD:.*]] = arith.addi %{{.*}}, %[[MUL]]
// CHECK: %[[OFF:.*]] = arith.index_cast %[[ADD]] : i32 to index
// CHECK: ktdp.construct_memory_view %[[OFF]], sizes: [4, 64], strides: [64, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<ct_local>} : memref<4x64xf16>
// CHECK-NOT: tts.pin
tt.func @affine_address(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %stride = arith.constant 512 : i32
  %base = arith.constant 4096 : i32
  %m = arith.muli %pid, %stride : i32
  %a = arith.addi %base, %m : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// Two consumers: one tile and one load EACH, both over the one view. Asserted
// because the alternative -- one load rewired to both uses -- would leave the two
// consumers sharing a value across what is now a memory boundary, which is the
// thing a pin exists to prevent.
// CHECK-LABEL: tt.func @two_consumers
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view
// CHECK: ktdp.store
// CHECK: %[[T1:.*]] = ktdp.construct_access_tile %[[VIEW]]
// CHECK: %[[L1:.*]] = ktdp.load %[[T1]]
// CHECK: %[[SQ:.*]] = math.sqrt %[[L1]]
// CHECK: %[[T2:.*]] = ktdp.construct_access_tile %[[VIEW]]
// CHECK: %[[L2:.*]] = ktdp.load %[[T2]]
// CHECK: arith.addf %[[L2]], %[[SQ]]
tt.func @two_consumers(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  %p = math.sqrt %e : tensor<4x64xf16>
  %q = arith.addf %e, %p : tensor<4x64xf16>
  tt.return %q : tensor<4x64xf16>
}

// -----
// A pin on a ktdp.load's result: the fixtures' "stage the input in the
// scratchpad" case, and the one place a pin buys something for a value that is
// already in memory. Two consumers then make one HBM read plus two scratchpad
// reads instead of two HBM reads. The GLOBAL view the load came from must survive
// untouched beside the new ct_local one, which is what the two memory_space
// checks together say.
// CHECK-LABEL: tt.func @stage_an_input
// CHECK: ktdp.construct_memory_view %{{.*}} {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>}
// CHECK: %[[X:.*]] = ktdp.load
// CHECK: ktdp.construct_memory_view %{{.*}} {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<ct_local>}
// CHECK: ktdp.store %[[X]]
#id = affine_map<(d0, d1) -> (d0, d1)>
#view = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>
tt.func @stage_an_input(%ptr: !tt.ptr<f16>) -> tensor<4x64xf16> {
  %c0 = arith.constant 0 : index
  %base = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %v = ktdp.construct_memory_view %base, sizes: [4, 64], strides: [64, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<4x64xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #view}
      : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %x = ktdp.load %t : <4x64xindex> -> tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  tts.pin %x, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  %y = math.sqrt %x : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A pinned function ARGUMENT, which is the same request as `@stage_an_input`
// above with the load moved out of the function: a `tensor` is a value whichever
// way it was defined, so it gets a buffer and its uses read it back. Nothing here
// looks at the producing op -- the store goes where the pin stood -- so a block
// argument needs no special case.
// CHECK-LABEL: tt.func @pin_an_argument
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view {{.*}} memory_space = #ktdp.memory_space<ct_local>
// CHECK: ktdp.store %arg0, %{{.*}}
// CHECK: %[[L:.*]] = ktdp.load
// CHECK: math.sqrt %[[L]]
tt.func @pin_an_argument(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %a = arith.constant 4096 : i32
  tts.pin %x, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  %y = math.sqrt %x : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// Rank 0, which a reduction produces. A rank-0 view and a rank-0 tile are both
// legal (buildRangeSetND has a rank-0 case, giving the always-true constraint),
// so the only thing that could go wrong here is the pass assuming a rank.
// CHECK-LABEL: tt.func @rank0
// CHECK: ktdp.construct_memory_view %{{.*}}, sizes: [], strides: [] {{.*}} : memref<f16>
// CHECK: ktdp.store
// CHECK: ktdp.load
tt.func @rank0(%x: tensor<f16>) -> tensor<f16> {
  %e = math.exp %x : tensor<f16>
  %a = arith.constant 0 : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<f16>
  %y = math.sqrt %e : tensor<f16>
  tt.return %y : tensor<f16>
}

// -----
// A pin with no consumer. The store still happens -- a pin is a materialization,
// not a hint -- and no load is built, because there is nothing to rewrite.
// CHECK-LABEL: tt.func @no_consumer
// CHECK: ktdp.store
// CHECK-NOT: ktdp.load
tt.func @no_consumer(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// No pin: the module comes back unchanged, and in particular gains no view. The
// pass places what a marker asked for and nothing else, so a kernel that pins
// nothing must be indistinguishable from one compiled before this pass existed.
// CHECK-LABEL: tt.func @unpinned
// CHECK-NOT: ktdp.construct_memory_view
// CHECK: math.sqrt
tt.func @unpinned(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// Two pins that do not overlap, at 0 and at 4096 elements. Asserted so the
// refusals in invalid.mlir read as a rule about intersection and not as a rule
// against pinning twice.
// CHECK-LABEL: tt.func @two_disjoint_pins
// CHECK-COUNT-2: ktdp.construct_memory_view {{.*}} memory_space = #ktdp.memory_space<ct_local>
// CHECK-NOT: tts.pin
tt.func @two_disjoint_pins(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %a0 = arith.constant 0 : i32
  tts.pin %e, address %a0 {memory_space = "ct_local"} : tensor<4x64xf16>
  %s = math.sqrt %e : tensor<4x64xf16>
  %a1 = arith.constant 4096 : i32
  tts.pin %s, address %a1 {memory_space = "ct_local"} : tensor<4x64xf16>
  %y = math.absf %s : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
