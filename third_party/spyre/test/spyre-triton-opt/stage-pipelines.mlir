// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s --check-prefix=KTIR
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=SYMBOLIC
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode="bind-base-addresses base-addresses=0,4294967296,8589934592" | FileCheck %s --check-prefix=BOUND
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir="grid=4" | FileCheck %s --check-prefix=GRID
// A prefix of its own, with nothing but NOT directives, so it scans the whole
// output. Interleaved with positive checks a NOT covers only the span between two
// of them, which for this claim would be the two lines ahead of the func.func.
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s --check-prefix=NOTT

// The two compile stages, each driven by one flag.
//
// Every other .mlir test here runs a single pass, which is why a reshuffle of a
// stage's *pass list* used to be invisible to the whole suite: there was nothing
// that ran a list. This runs both, so a pass added, dropped or moved between the
// stages shows up as a diff somewhere below.
//
// The claims are the ones no single pass makes, and they are deliberately about
// the shape of the output rather than its every line -- a pass's own behaviour is
// pinned by its own test, and restating it here would make this file fail for
// reasons that have nothing to do with the sequence.

module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %n = arith.constant 1024 : i32
    %s = arith.constant 1 : i64
    %pid = tt.get_program_id x : i32
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%n], [%s] : <f32>, <1024xf32>
    %y_desc = tt.make_tensor_descriptor %y_ptr, [%n], [%s] : <f32>, <1024xf32>
    %o_desc = tt.make_tensor_descriptor %out_ptr, [%n], [%s] : <f32>, <1024xf32>
    %off = arith.muli %pid, %n : i32
    %x = tt.descriptor_load %x_desc[%off] : !tt.tensordesc<1024xf32> -> tensor<1024xf32>
    %y = tt.descriptor_load %y_desc[%off] : !tt.tensordesc<1024xf32> -> tensor<1024xf32>
    %r = arith.addf %x, %y : tensor<1024xf32>
    tt.descriptor_store %o_desc[%off], %r : !tt.tensordesc<1024xf32>, tensor<1024xf32>
    tt.return
  }
}

// Nothing of the tt dialect reaches the end of the stage -- the whole claim of
// its name, and the reason ConvertFunctions has to run after every pass that
// consumes a !tt.ptr.
//
// NOTT-NOT: tt.func
// NOTT-NOT: tt.ptr
// NOTT-NOT: tt.descriptor
// NOTT-NOT: tt.tensordesc
// NOTT-NOT: tt.get_program_id
// NOTT-NOT: tt.get_num_programs
// NOTT-NOT: tt.return

// The `ktir` stage, whole. The entry point is a func.func over `index`,
// DistributeWork has stamped its grid and replaced the program id, and the
// elementwise add has become a linalg.generic whose `outs` is a fresh
// tensor.empty rather than one of its own `ins` -- that last one is the property
// dbo-opt needs, and the stage's pass list is what arranges it.
//
// KTIR-LABEL:   func.func @add_kernel(
// KTIR-SAME:      %[[X:.*]]: index, %[[Y:.*]]: index, %[[OUT:.*]]: index
// KTIR-SAME:      attributes {grid = [32]} {
// KTIR:           %[[TILE_ID:.*]] = ktdp.get_compute_tile_id : index
// KTIR:           ktdp.construct_memory_view %[[X]],
// KTIR:           %[[XT:.*]] = ktdp.load
// KTIR:           %[[YT:.*]] = ktdp.load
// KTIR:           %[[EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// KTIR:           %[[SUM:.*]] = linalg.generic
// KTIR-SAME:        ins(%[[XT]], %[[YT]] : tensor<1024xf32>, tensor<1024xf32>)
// KTIR-SAME:        outs(%[[EMPTY]] : tensor<1024xf32>)
// KTIR:             arith.addf
// KTIR:           ktdp.store %[[SUM]],
// KTIR:           return

// The `spyrecode` stage in the default argument-passing mode. It is a KTIR → KTIR
// round trip: the pointer arguments survive, because the addresses are not known
// at compile time and the runtime patches them in.
//
// SYMBOLIC-LABEL:   func.func @add_kernel(
// SYMBOLIC-SAME:      %{{.*}}: index, %{{.*}}: index, %{{.*}}: index
// SYMBOLIC:           ktdp.store

// And in the binding mode, which is the one genuine choice in that stage: the
// arguments become constants and leave the signature, because the dataflow
// scheduler requires a zero-argument entry function. The canonicalize and CSE
// that follow the materialization have folded each address into its memory view.
//
// BOUND-LABEL:   func.func @add_kernel() attributes {grid = [32]} {
// BOUND:           %[[ADDR0:.*]] = arith.constant 0 : index
// BOUND:           %[[ADDR1:.*]] = arith.constant 4294967296 : index
// BOUND:           %[[ADDR2:.*]] = arith.constant 8589934592 : index
// BOUND:           ktdp.construct_memory_view %[[ADDR0]],
// BOUND:           ktdp.construct_memory_view %[[ADDR1]],
// BOUND:           ktdp.construct_memory_view %[[ADDR2]],

// The grid reaches DistributeWork through the pipeline's own option, so the one
// compile input this stage takes is drivable from the command line too.
//
// GRID-LABEL:   func.func @add_kernel(
// GRID-SAME:      attributes {grid = [4]} {
