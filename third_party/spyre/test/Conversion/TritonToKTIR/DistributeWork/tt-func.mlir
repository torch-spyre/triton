// NOTE: Assertions were generated with utils/generate-test-checks.py, then
// hand-extended with the negative lines called out per case.

// RUN: spyre-triton-opt %s -split-input-file --distribute-work-to-spyre-cores='grid=32,1' | FileCheck %s

// The pass runs on tt.func directly, with no --convert-ttir-functions in front
// of it (contrast every other file here, which converts first).
//
// That works because the pass walks tt.get_program_id directly and finds the
// enclosing function through FunctionOpInterface rather than by matching
// func.func. So ConvertFunctions may run before it, after it, or not at all.
// Two consequences show up in the labels below: the function stays tt.func, and
// the pointer argument keeps its !tt.ptr type because nothing retyped it. Both
// are load-bearing -- a pass that quietly required func.func would leave these
// kernels untouched, and the grid attribute would be missing.
//
// Grid is [32, 1], so both kernels read two axes; a 1-D kernel under this grid
// would be a rank mismatch rather than a tt.func test.

// -----
// A 2-D kernel inside tt.func: the pid rewrite, the two-result variadic op, and
// the grid attribute all land on tt.func exactly as they do on func.func.

// CHECK-LABEL:   tt.func @multi_axis_on_tt_func(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>) attributes {grid = [32, 1]} {
// CHECK:           %[[VAL_1:.*]]:2 = ktdp.get_compute_tile_id : index, index{{$}}
// CHECK-NOT:       ktdp.get_compute_tile_id
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]]#0 : index to i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]]#1 : index to i32
// CHECK-NOT:       arith.index_cast
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK-NOT:       tt.get_program_id
// CHECK:           tt.return
// CHECK:         }
tt.func @multi_axis_on_tt_func(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %sum = arith.addi %px, %py : i32
  tt.return
}

// -----
// A second tt.func kernel whose pids feed nothing. The rewrite and the grid
// stamp do not depend on the pid having a consumer.

// CHECK-LABEL:   tt.func @runs_on_tt_func(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>) attributes {grid = [32, 1]} {
// CHECK:           %[[VAL_1:.*]]:2 = ktdp.get_compute_tile_id : index, index{{$}}
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]]#0 : index to i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]]#1 : index to i32
// CHECK-NOT:       tt.get_program_id
// CHECK:           tt.return
// CHECK:         }
tt.func @runs_on_tt_func(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  tt.return
}
