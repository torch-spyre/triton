// NOTE: Assertions were generated with utils/generate-test-checks.py, then
// hand-extended with the negative lines called out per case.

// RUN: spyre-triton-opt %s -split-input-file --convert-ttir-functions --distribute-work-to-spyre-cores='grid=16,2' | FileCheck %s

// Tests for the tt.get_num_programs fold on a 2-D grid, grid=[16, 2].
//
// tt.get_num_programs on axis i folds to arith.constant grid[i] : i32, using the
// same grid the pass stamps on the function. The two axes here have different
// sizes so each fold is distinguishable: axis 0 -> 16, axis 1 -> 2. A test on a
// square grid could not tell the two folds apart.
//
// The 1-D fold lives in distribute-work.mlir; this file adds the per-axis case
// and the rule that a num_programs read counts toward the kernel's
// dimensionality just as a program_id read does.

// -----
// Both axes read pid and num_programs. Each muli pairs a pid with the folded
// constant for its own axis, so the two folds are read off the two muli ops --
// the SSA captures below tie constant 16 to the axis-0 muli and constant 2 to
// the axis-1 one, which is what pins the folded *values* rather than merely the
// presence of two constants.

// CHECK-LABEL:   func.func @num_programs_fold_2d(
// CHECK-SAME:  %[[VAL_0:.*]]: index) attributes {grid = [16, 2]} {
// CHECK:           %[[VAL_1:.*]]:2 = ktdp.get_compute_tile_id : index, index
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]]#0 : index to i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]]#1 : index to i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_3]], %[[VAL_5]] : i32
// CHECK-NOT:       tt.get_num_programs
// CHECK-NOT:       tt.get_program_id
// CHECK:           return
// CHECK:         }
tt.func @num_programs_fold_2d(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %nx = tt.get_num_programs x : i32
  %ny = tt.get_num_programs y : i32
  %a = arith.muli %px, %nx : i32
  %b = arith.muli %py, %ny : i32
  tt.return
}

// -----
// num_programs alone bumps the dimensionality. This kernel reads pid on axis 0
// only, but num_programs on axis 1 -- both ops contribute, so dimensionality is
// 2, the tile-id op has two results, and the fold picks grid[1] = 2. A 1-D grid
// here would be the rank mismatch pinned in distribute-work-invalid.mlir.
//
// The constant is captured and tied to the muli, so a fold that reached for
// grid[0] = 16 instead would break the match rather than pass on the mere
// presence of a constant.

// CHECK-LABEL:   func.func @num_programs_bumps_dimensionality(
// CHECK-SAME:  %[[VAL_0:.*]]: index) attributes {grid = [16, 2]} {
// CHECK:           %[[VAL_1:.*]]:2 = ktdp.get_compute_tile_id : index, index{{$}}
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]]#0 : index to i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]]#1 : index to i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_2]], %[[VAL_4]] : i32
// CHECK-NOT:       tt.get_num_programs
// CHECK-NOT:       tt.get_program_id
// CHECK:           return
// CHECK:         }
tt.func @num_programs_bumps_dimensionality(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %ny = tt.get_num_programs y : i32
  %r = arith.muli %px, %ny : i32
  tt.return
}
