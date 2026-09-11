// RUN: spyre-triton-opt %s --convert-ttir-functions --distribute-work-to-spyre-cores='grid=32' -split-input-file -verify-diagnostics

// Inputs --distribute-work-to-spyre-cores rejects, at grid=[32].
//
// Two of the pass's three preconditions are pinned here; the third (dense axes)
// needs a 3-element grid and lives in distribute-work-invalid-3d.mlir, and the
// symmetric rank mismatch needs a 2-element grid and lives in
// distribute-work-invalid-2d.mlir. Grid is a pass option, so one grid per file.
//
// Each annotation below matches by substring on the diagnostic attached to the
// following line. The strings are deliberately specific: a loose substring
// would happily accept a different one of the three diagnostics, which is the
// exact confusion these tests exist to prevent. -split-input-file keeps one
// rejection from masking the next.

// -----
// The caller supplied a 1-D grid, but the kernel reads two program-id axes.
// There is no coherent way to partition a 2-D kernel across a 1-D grid, so the
// pass refuses rather than guess. The message must name the caller's rank, the
// kernel's dimensionality, and the axis span the kernel actually reads.
// expected-error @below {{DistributeWork: grid rank 1 does not match kernel's pid dimensionality 2 (function reads tt.get_program_id / tt.get_num_programs on axes 0..1)}}
tt.func @grid_rank_too_short(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  tt.return
}

// -----
// A kernel that reads num_programs but never program_id. It asks how many
// programs there are while never locating itself in the grid, so no per-core
// branch can use the answer -- almost certainly a bug, flagged rather than
// silently lowered into unreachable code.
//
// Triton source pattern:
//
//   # Missing tl.program_id -- tl.num_programs alone has nothing to act on
//   num_cores = tl.num_programs(0)
//   result = do_something(num_cores)  # per-core location is unknown
//
// expected-error @below {{DistributeWork: function reads tt.get_num_programs but never tt.get_program_id; a kernel that asks for the grid size without locating itself in the grid is almost certainly a bug}}
tt.func @num_programs_without_pid(%arg0: !tt.ptr<f32>) {
  %n = tt.get_num_programs x : i32
  tt.return
}
