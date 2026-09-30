// NOTE: Assertions were generated with utils/generate-test-checks.py, then
// hand-extended with the negative and arity lines called out per case.

// RUN: spyre-triton-opt %s -split-input-file --convert-ttir-functions --distribute-work-to-spyre-cores='grid=1,1,32' | FileCheck %s

// Tests for --distribute-work-to-spyre-cores on a 3-D grid, grid=[1, 1, 32].
//
// A kernel reading axis z has dimensionality 3 (numDims = max axis + 1), so the
// caller must pass a 3-element grid and the variadic tile-id op yields three
// index results. The casts for axes 0 and 1 are emitted even though this grid
// puts a single core on each -- all 32 cores partition axis z.

// -----
// All three axes are read. x and y must be read even though the grid is 1x1x32,
// because the pass requires the axes read to be dense from 0: skipping axis 1
// would trip that invariant, which distribute-work-invalid-3d.mlir pins.
//
// Three casts and a three-entry result list are the whole claim here; the
// trailing {{$}} stops a fourth result from matching as a prefix.

// CHECK-LABEL:   func.func @axis_z(
// CHECK-SAME:  %[[VAL_0:.*]]: index) attributes {grid = [1, 1, 32]} {
// CHECK:           %[[VAL_1:.*]]:3 = ktdp.get_compute_tile_id : index, index, index{{$}}
// CHECK-NOT:       ktdp.get_compute_tile_id
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]]#0 : index to i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]]#1 : index to i32
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_1]]#2 : index to i32
// CHECK-NOT:       arith.index_cast
// CHECK-NOT:       tt.get_program_id
// CHECK:           return
// CHECK:         }
tt.func @axis_z(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %pz = tt.get_program_id z : i32
  tt.return
}
