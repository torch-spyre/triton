// NOTE: Assertions were generated with utils/generate-test-checks.py, then
// hand-extended with the negative and arity lines called out per case.

// RUN: spyre-triton-opt %s -split-input-file --convert-ttir-functions --distribute-work-to-spyre-cores='grid=32,1' | FileCheck %s

// Tests for --distribute-work-to-spyre-cores on a 2-D grid, grid=[32, 1].
//
// The grid is a pass option rather than a per-case one, so this file exists
// purely to carry a different grid rank than distribute-work.mlir. The grid list
// is stamped verbatim on the function, so its rank must equal the kernel's pid
// dimensionality -- see distribute-work-invalid{,-2d,-3d}.mlir for the ways that
// can go wrong.

// -----
// A kernel reading both program_id x and program_id y gets ONE variadic
// ktdp.get_compute_tile_id returning two index values (one per grid dim) and
// one arith.index_cast per axis to recover the i32 program id.
//
// Triton source pattern:
//
//   pid_x = tl.program_id(0)
//   pid_y = tl.program_id(1)
//   # both axes share one underlying tile-id op after lowering
//   row_offset = pid_x * BLOCK_M
//   col_offset = pid_y * BLOCK_N
//
// The ": index, index" result-type list is the arity claim. The trailing {{$}}
// pins it to end-of-line, so a third result would break the match rather than
// matching as a prefix. The absence lines sit between the last matched op and
// the return: placed after the closing brace they would guard only the empty
// tail of the module and could never fire.

// CHECK-LABEL:   func.func @multi_axis_pid(
// CHECK-SAME:  %[[VAL_0:.*]]: index) attributes {grid = [32, 1]} {
// CHECK:           %[[VAL_1:.*]]:2 = ktdp.get_compute_tile_id : index, index{{$}}
// CHECK-NOT:       ktdp.get_compute_tile_id
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]]#0 : index to i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]]#1 : index to i32
// CHECK-NOT:       arith.index_cast
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK-NOT:       tt.get_program_id
// CHECK:           return
// CHECK:         }
tt.func @multi_axis_pid(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %sum = arith.addi %px, %py : i32
  tt.return
}
