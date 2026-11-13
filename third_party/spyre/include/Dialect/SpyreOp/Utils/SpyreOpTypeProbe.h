//===- SpyreOpTypeProbe.h - Which types a spyreop rule lowers -------------===//
//
// probeSpyreOpRule: asks the spyreop ops a rule would build whether they take
// the candidate's types, by building them and running their verifiers.
//
// WHY A PROBE. What types a spyreop op takes is written once, in its op
// definition (SpyreOp.td in ktir-mlir-frontend), and MLIR turns it into a
// verifier. It generates nothing that answers "does this op take f64?" without
// an op to ask about. The choices were:
//
//   - copy the type lists into this repository, as a predicate per rule. They
//     go stale silently when the definition changes, e.g. when bf16 is added;
//   - build the op and ask its verifier, which cannot disagree with the
//     definition because it is the definition.
//
// This takes the second. It is the technique of verifySpyreIntrinsicCall,
// which checks a `tl.spyre_op` call the same way. MLIR has no named utility
// for it; the pieces are standard (a detached Block, mlir::verify, a scoped
// diagnostic handler).
//
// WHAT IT COSTS. A few ops are created and destroyed per rule tried.
// LowerSpyreOps tries a rule per op, and FuseGenerics tries the rules with a
// reader per compare-and-reader pair it considers, so the cost is bounded by
// the number of such pairs. Memoizing on (rule, operand types, attributes) is
// possible if it ever shows in a profile.
//
// WHAT IT REQUIRES OF A RULE. The probe runs `SpyreOpLoweringRule::build` into
// a scratch block, so `build` must read its operands only from the `operands`
// it is given, and the candidate only for types and attributes. The probe
// checks this rather than trusting it: a build that reaches a value outside the
// scratch block, or returns a value of another type than the op it replaces,
// aborts, naming the rule. That is a bug in the rule, and a broken rule table
// is a compiler bug rather than input, as in verifySpyreOpLoweringRules. A
// diagnostic the build itself emits is dropped: the reason is a verifier's.
//
// WHAT IT DOES NOT SEE. A verifier knows types and op-local invariants. It does
// not know that a reciprocal's numerator must be 1.0 or that a predicate has a
// counterpart: those are the rule's `guard`, which runs before the probe. A
// dialect may also admit what the device does not; the backend refuses those.
//
// WHERE IT RUNS. After ruleShapeMatches, at both places that ask whether a rule
// lowers a candidate: LowerByRule before it lowers, and matchLowerableGroup
// before FuseGenerics fuses. Both call the two steps by name. The probe takes
// the ShapeCheckedRule that ruleShapeMatches makes, so it is never asked about
// a rule whose shape the candidate does not have.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPTYPEPROBE_H
#define TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPTYPEPROBE_H

#include "Dialect/SpyreOp/Utils/SpyreOpLoweringRules.h"

#include <string>

namespace mlir::triton::spyre {

/// Whether every op `shape.rule().build` makes for `shape.candidate()` passes
/// its own verifier, on operands of the types `shape.operands()` has.
///
/// The build runs into a detached block whose arguments stand for the
/// operands, through a builder that notifies no rewriter, so no real IR is
/// touched; the block and everything in it are destroyed before returning. The
/// verifiers' diagnostics are caught by a handler scoped to the call, so
/// nothing is reported, and the first becomes `why`.
bool probeSpyreOpRule(const ShapeCheckedRule &shape, std::string &why);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPTYPEPROBE_H
