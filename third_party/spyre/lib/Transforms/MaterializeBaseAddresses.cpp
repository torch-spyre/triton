//===- MaterializeBaseAddresses.cpp - Inline fixed HBM base addresses -----===//
//
// Replaces base-address function arguments with `arith.constant ... : index`
// in the entry block and drops them from the signature.
//
// The dataflow scheduler consumes a `func.func` whose base addresses are
// materialized constants rather than arguments:
//
//   Before (post-ConvertFunctions):        After (addresses [1024, 12288]):
//     func.func @add(%a: index,             func.func @add() {
//                    %b: index) {             %0 = arith.constant 1024 : index
//       ... %a ... %b ...                     %1 = arith.constant 12288 : index
//                                             ... %0 ... %1 ...
//
// This is an **opt-in** mode for scheduler testing. The pass is not
// in _CORE_PIPELINE_PASSES nor in the fused add_convert_ttir_to_ktdp binding;
// it is reached only via the `required_fixes` mechanism in
// third_party/spyre/backend/compiler.py. With no addresses supplied it is a
// no-op, so the default argument-passing path is unchanged.
//
// Indexing: the address list is positional over the entry block's **`index`
// arguments**, not over all arguments. ConvertFunctions retypes only
// `!tt.ptr` arguments to `index` and leaves every other type alone, so after
// it runs "`index` argument" and "was a pointer argument" are the same set.
// Scanning `index` args therefore makes an interleaved signature such as
// `(%a: index, %n: i32, %b: index)` behave correctly — %a and %b get the two
// addresses, and the i32 runtime scalar %n is never a candidate.
//
// Positional keying is the only option available: ConvertFunctions clones the
// function from its FunctionType and copies only visibility, so argument names
// and all argument attributes are already gone by the time this pass runs.
//
// Note that the address list is indexed by ordinal `i` among `index` args,
// while getArgument/eraseArguments take a signature position `pos`. The two
// are different whenever a non-`index` argument precedes a base address, so
// `indexArgPositions` keeps them explicitly separate below.
//
// Units are ELEMENTS, not bytes. `$offset` on ktdp.construct_memory_view feeds
// MemRef.base_ptr, which ktir-cpu documents as "an element index — the number of
// elements from the start of the address space, matching what MLIR pointer
// operands carry" (ktir_cpu/ir_types.py:47-56); the byte position is derived as
// `base_ptr * bytes_per_elem(dtype)` (ir_types.py:84-87). The `sizes`/`strides`
// on this op are element counts for the same reason.
//
// The ODS prose calls $offset a "start address" without stating units, so the
// interpreter is authoritative. The scheduler's round-trip.mlir agrees
// arithmetically: its memref<96x64xf16> views based at 1024/12288/18432 span
// 6144 elements each and tile exactly, whereas read as bytes they would need
// 12288 each and overlap.
//
// The pass does no scaling — it emits the integer it is given. Scaling could not
// be done here in any case: element size is per-descriptor, and this pass sees
// only the function signature.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_MATERIALIZEBASEADDRESSES
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

struct MaterializeBaseAddressesPass
    : public mlir::triton::spyre::impl::MaterializeBaseAddressesBase<
          MaterializeBaseAddressesPass> {

  using MaterializeBaseAddressesBase::MaterializeBaseAddressesBase;

  MaterializeBaseAddressesPass(ArrayRef<int64_t> addrs) {
    // ListOption inherits from llvm::cl::list which accepts ArrayRef
    // via operator=.
    baseAddresses = addrs;
  }

  void runOnOperation() override {
    // Opt-in: with no addresses supplied the pass leaves the IR alone.
    if (baseAddresses.empty())
      return;

    // Entry-point functions only. A private helper is not a kernel boundary,
    // and its arguments are supplied by its caller, not by the runtime.
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      if (!funcOp.isPublic())
        continue;
      if (failed(materialize(funcOp)))
        return signalPassFailure();
    }
  }

private:
  LogicalResult materialize(func::FuncOp funcOp) {
    Block &entry = funcOp.getBody().front();

    // Step 1: positions (in the signature) of every `index` argument, in
    // scan order. Entry i of the address list belongs to indexArgPositions[i].
    SmallVector<unsigned> indexArgPositions;
    for (unsigned pos = 0, e = entry.getNumArguments(); pos < e; ++pos)
      if (entry.getArgument(pos).getType().isIndex())
        indexArgPositions.push_back(pos);

    // Step 2a: more addresses than pointers has no correct reading — the
    // caller named base addresses the kernel does not have. A shorter list is
    // legal: it materializes a prefix and leaves the rest as arguments, which
    // is well-defined and useful for incremental bring-up.
    if (baseAddresses.size() > indexArgPositions.size()) {
      funcOp.emitError()
          << "MaterializeBaseAddresses: " << baseAddresses.size()
          << " base addresses supplied but the function has only "
          << indexArgPositions.size()
          << " index argument(s) to materialize them into";
      return failure();
    }

    // Step 2b: int64_t admits negatives but an address cannot be negative;
    // reject rather than silently emit `arith.constant -8 : index`.
    for (auto [i, addr] : llvm::enumerate(baseAddresses)) {
      if (addr < 0) {
        funcOp.emitError()
            << "MaterializeBaseAddresses: base address " << i
            << " is negative (" << addr << "); addresses must be >= 0";
        return failure();
      }
    }

    // Step 3: create every constant *before* erasing any argument, so no
    // argument position shifts while we are still reading positions.
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&entry);

    BitVector toErase(entry.getNumArguments());
    for (unsigned i = 0, e = baseAddresses.size(); i < e; ++i) {
      unsigned pos = indexArgPositions[i];
      Value cst = arith::ConstantIndexOp::create(builder, funcOp.getLoc(),
                                                 baseAddresses[i]);
      entry.getArgument(pos).replaceAllUsesWith(cst);
      toErase.set(pos);
    }

    // Step 4: one erase call, so surviving arguments shift correctly in a
    // single step and keep their relative order.
    entry.eraseArguments(toErase);

    // Step 5: rebuild the signature from the arguments that survived.
    SmallVector<Type> newArgTypes(entry.getArgumentTypes());
    funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes,
                                     funcOp.getFunctionType().getResults()));
    return success();
  }
};

} // namespace

namespace mlir::triton::spyre {
std::unique_ptr<OperationPass<ModuleOp>>
createMaterializeBaseAddressesPass(ArrayRef<int64_t> baseAddresses) {
  return std::make_unique<MaterializeBaseAddressesPass>(baseAddresses);
}
} // namespace mlir::triton::spyre
