// RUN: spyre-triton-opt %s --convert-ttir-functions --distribute-work-to-spyre-cores='grid=32,1' -split-input-file -verify-diagnostics

// The rank mismatch in the other direction, at grid=[32, 1].
//
// Symmetric to @grid_rank_too_short in distribute-work-invalid.mlir. It needs
// its own file because the grid rank is a pass option and the two cases require
// different ones.

// -----
// The caller declared more grid axes than the kernel reads. Accepting this
// would leave an axis silently unused, so the pass refuses and names both
// numbers plus the fact that the kernel only touches axis 0.
// expected-error @below {{DistributeWork: grid rank 2 does not match kernel's pid dimensionality 1 (function reads tt.get_program_id / tt.get_num_programs on axes 0..0)}}
tt.func @grid_rank_too_long(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  tt.return
}
