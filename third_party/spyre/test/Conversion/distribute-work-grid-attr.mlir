// RUN: spyre-triton-opt %s --convert-ttir-functions --distribute-work-to-spyre-cores='grid=1' | FileCheck %s --check-prefix=GRID1
// RUN: spyre-triton-opt %s --convert-ttir-functions --distribute-work-to-spyre-cores='grid=32' | FileCheck %s --check-prefix=GRID32
// RUN: spyre-triton-opt %s --convert-ttir-functions --distribute-work-to-spyre-cores='grid=64' | FileCheck %s --check-prefix=GRID64

// The grid attribute is the option list, verbatim.
//
// One kernel, three grids. Because the grid is a pass option and not a per-case
// one, the parametrization lives in the RUN lines above -- one per grid, each
// with its own prefix -- rather than in three copies of the kernel.
//
// The kernel reads only axis 0, so every grid here is a 1-element list: the
// rank has to match the kernel's pid dimensionality or the pass rejects it.
//
// These are hand-written rather than generated, because the generator emits a
// single prefix and would collapse the three cases into one.
//
// Each prefix asserts its own grid on the -SAME line. A negative line per
// prefix ("not the other two grids") was tried and removed: a run invoked with
// grid=1 can never print grid = [32], so such a line cannot fail and would be
// decoration. The -SAME lines already discriminate, since each names an exact
// bracketed list.

// GRID1-LABEL:  func.func @grid_reflects_option
// GRID1-SAME:   attributes {grid = [1]}

// GRID32-LABEL: func.func @grid_reflects_option
// GRID32-SAME:  attributes {grid = [32]}

// GRID64-LABEL: func.func @grid_reflects_option
// GRID64-SAME:  attributes {grid = [64]}
tt.func @grid_reflects_option(%arg0: !tt.ptr<f32>) {
  %pid = tt.get_program_id x : i32
  tt.return
}
