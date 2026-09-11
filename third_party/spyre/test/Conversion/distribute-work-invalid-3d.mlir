// RUN: spyre-triton-opt %s --convert-ttir-functions --distribute-work-to-spyre-cores='grid=32,1,1' -split-input-file -verify-diagnostics

// The dense-axes precondition, at grid=[32, 1, 1].
//
// Its own file because it needs a 3-element grid: the kernel below reads axis 2,
// so its dimensionality is 3 and any shorter grid would trip the rank check
// first and never reach the density check this test is about.

// -----
// The kernel reads axes 0 and 2 but skips axis 1. Dense indexing from 0 is
// required so each axis's partition has one continuous meaning; a gap would
// leave axis 1 unused while the author believes work is spread over three axes.
// The message must name the skipped axis and the highest axis read, so the gap
// is visible.
// expected-error @below {{DistributeWork: function reads grid axes non-densely (axis 1 is skipped; highest axis read is 2)}}
tt.func @axes_non_dense(%arg0: !tt.ptr<f32>) {
  %px = tt.get_program_id x : i32
  %pz = tt.get_program_id z : i32
  tt.return
}
