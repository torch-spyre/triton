// RUN: spyre-triton-opt %s --materialize-pinned-buffers="lx-capacity-bytes=-1" -verify-diagnostics

// A NEGATIVE capacity, refused. It used to behave exactly like the 0 that means "do
// not ask", because the capacity test read `capacity > 0`, so `lx-capacity-bytes=-1`
// turned the rule off and a pin past any scratchpad passed.
//
// Its own file, and two things about it are the reason. The diagnostic is on the
// MODULE rather than on a pin, because an option is wrong independently of what it is
// handed and a module with no pins at all must not launder one; and a module-level
// error needs an explicit `module` to be anchored to, which the other files in this
// directory do not have. The RUN line here is the whole of the case: there is nothing
// about the input that makes it fail.
//
// Why not clamp or ignore it: a negative budget is not a budget, and the two readings
// available -- "unlimited" and "nothing fits" -- are opposites. Neither is worth
// guessing on the caller's behalf.

// expected-error @+1 {{lx-capacity-bytes must not be negative, and this one is -1}}
module {
  tt.func @pinned(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
    %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
    %y = math.sqrt %e : tensor<4x64xf16>
    tt.return %y : tensor<4x64xf16>
  }
}
