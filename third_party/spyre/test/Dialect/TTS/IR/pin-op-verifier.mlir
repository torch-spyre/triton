// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// The tts.pin OP's verifier. No pass runs here.
//
// Every rule below is STRUCTURAL -- what shape an address expression may have,
// not what numbers it may hold. The numeric rules (a stick-aligned offset, a
// range that fits the scratchpad, ranges that do not overlap) need the launch
// grid and the device description, neither of which an op can see, so they live
// in PlacePinnedValues and are tested with it.
//
// The division matters for a reader deciding where a new rule belongs: if it can
// be answered from the op alone it goes here, and if it needs a pass option it
// does not.

// The memory-space vocabulary is ktdp's MemorySpaceKind, reached through
// `symbolizeMemorySpaceKind` so the two names are never restated in C++.
tt.func @unknown_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{unknown memory space 'lx': expected 'ct_local'}}
  tts.pin %e {memory_space = "lx"} : tensor<4x64xf16>
  tt.return
}

// -----
// Case matters, since the enum's own spelling is lower case.
tt.func @wrong_case(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{unknown memory space 'CT_LOCAL'}}
  tts.pin %e {memory_space = "CT_LOCAL"} : tensor<4x64xf16>
  tt.return
}

// -----
// `global` is a KNOWN kind and still not pinnable, so it gets its own message
// rather than being reported as a misspelling. lx-placement.md's first
// assumption puts HBM intermediates outside a pin: one is written as a
// tl.make_tensor_descriptor with an explicit store and load, and nothing here
// allocates an anonymous device buffer. Refused by the op rather than by the
// pass, so the diagnostic is about the space the author chose and not about a
// missing address.
tt.func @global_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space 'global' cannot be pinned: only 'ct_local' is}}
  tts.pin %e {memory_space = "global"} : tensor<4x64xf16>
  tt.return
}

// -----
// And with an address too, so the message is the space's either way rather than
// changing depending on what else the pin carries.
tt.func @global_with_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  // expected-error @+1 {{memory space 'global' cannot be pinned}}
  tts.pin %e, address %a {memory_space = "global"} : tensor<4x64xf16>
  tt.return
}

// -----
// An address read at run time forfeits the whole point of the restriction: its
// address set is not enumerable, so neither capacity nor disjointness can be
// answered about it.
tt.func @runtime_address(%x: tensor<4x64xf16>, %k: i32) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{address must be a constant or `base + tl.program_id(0) * stride` with constant coefficients}}
  tts.pin %e, address %k {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// A non-constant COEFFICIENT is the same problem one level down: the shape is
// affine in the program id, but the set it describes is not known until run
// time.
tt.func @runtime_stride(%x: tensor<4x64xf16>, %k: i32) {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %m = arith.muli %pid, %k : i32
  // expected-error @+1 {{address must be a constant or `base + tl.program_id(0) * stride`}}
  tts.pin %e, address %m {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// Axis X only. The grid this backend distributes over is one-dimensional, so X
// is the only axis whose address set has prod(grid) as a bound to enumerate
// against -- a Y term is well formed arith and still unenumerable here.
tt.func @program_id_y(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id y : i32
  %stride = arith.constant 256 : i32
  %m = arith.muli %pid, %stride : i32
  // expected-error @+1 {{address must be a constant or `base + tl.program_id(0) * stride`}}
  tts.pin %e, address %m {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// Two program-id terms, and the one refusal here that is CONSERVATIVE rather
// than principled. `pid*256 + pid*512` is `pid*768`, an admissible set; the
// matcher refuses it only because it does not sum strides. What it must not do
// is match one side and ignore the other, which would leave the pin occupying a
// range nobody computed -- and summing would avoid that too. Refusing is the
// cheaper of the two, and nothing in tree spells two terms: the frontend folds
// tl.constexpr coefficients in Python, so BASE + pid*STRIDE arrives as exactly
// one multiply and one add. Relax this to a sum if a caller ever needs it.
tt.func @two_pid_terms(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %s1 = arith.constant 256 : i32
  %s2 = arith.constant 512 : i32
  %m1 = arith.muli %pid, %s1 : i32
  %m2 = arith.muli %pid, %s2 : i32
  %a = arith.addi %m1, %m2 : i32
  // expected-error @+1 {{address must be a constant or `base + tl.program_id(0) * stride`}}
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// A dynamic extent cannot be placed: there is no buffer size, so no capacity
// answer and nothing to build a view over. Refused by the operand's type
// constraint rather than by a hand-written rule.
tt.func @dynamic_shape(%x: tensor<?x64xf16>) {
  %e = math.exp %x : tensor<?x64xf16>
  %a = arith.constant 4096 : i32
  // expected-error @+1 {{operand #0 must be statically shaped tensor of any type values}}
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<?x64xf16>
  tt.return
}

// The forms that DO verify, so the rejections above are read as rules and not as
// the op being hard to satisfy. Commutativity on both operators, and the two
// degenerate coefficients.
//
// Note there are no rules of dashes anywhere in this file, and no comment quotes
// the split marker either: -split-input-file matches the marker as a substring,
// so both would start a new chunk and the text after them would be parsed as IR.

// -----
tt.func @accepted_forms(%x: tensor<4x64xf16>) {
  %e0 = math.exp %x : tensor<4x64xf16>
  %e1 = math.exp %x : tensor<4x64xf16>
  %e2 = math.exp %x : tensor<4x64xf16>
  %e3 = math.exp %x : tensor<4x64xf16>
  %e4 = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %stride = arith.constant 256 : i32
  %base = arith.constant 4096 : i32

  // A bare constant: stride 0.
  tts.pin %e0, address %base {memory_space = "ct_local"} : tensor<4x64xf16>

  // A bare program id: base 0, stride 1.
  tts.pin %e1, address %pid {memory_space = "ct_local"} : tensor<4x64xf16>

  // pid * stride, with the constant on either side of the multiply.
  %m = arith.muli %pid, %stride : i32
  %n = arith.muli %stride, %pid : i32
  tts.pin %e2, address %m {memory_space = "ct_local"} : tensor<4x64xf16>
  tts.pin %e3, address %n {memory_space = "ct_local"} : tensor<4x64xf16>

  // base + pid * stride, with the base on the right of the add.
  %a = arith.addi %m, %base : i32
  tts.pin %e4, address %a {memory_space = "ct_local"} : tensor<4x64xf16>

  tt.return
}
