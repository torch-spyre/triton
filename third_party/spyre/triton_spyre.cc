//===- triton_spyre.cc - Pybind11 bindings for Spyre backend --------------===//
//
// Exposes the KTDP dialect and Spyre lowering passes to Python via pybind11.
// This is compiled as part of the main libtriton shared library.
// The entry point init_triton_spyre() is called from main.cc via the
// FOR_EACH_P(INIT_BACKEND, ...) macro.
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

/// Returns true when elementwise tensor arith/math should be converted to
/// linalg.generic by upstream MLIR's convert-elementwise-to-linalg pass.
///
/// Read from the `SPYRE_LINALG_COMPUTE` environment variable, which accepts
/// "0", "false", or "off" (case-insensitive) to disable. Any other value, and
/// being unset, enables it. Enabled is the default because the Spyre dataflow
/// scheduler accepts only linalg ops: KTIR that still carries tensor-typed
/// arith cannot be scheduled. The off switch exists to bisect a miscompile by
/// diffing KTIR with the conversion on against KTIR with it off for the same
/// kernel.
///
/// Read once per process. The environment is not expected to change during a
/// compilation, and re-reading per call would let two kernels compiled in one
/// process disagree.
///
/// This setting is NOT part of `SpyreOptions.hash()` in
/// `third_party/spyre/backend/compiler.py`, so cached compilation artifacts do
/// not distinguish the two settings. Clear the Triton cache when toggling it.
bool elementwiseToLinalgEnabled() {
  static const bool enabled = [] {
    const char *raw = std::getenv("SPYRE_LINALG_COMPUTE");
    if (!raw)
      return true;
    llvm::StringRef value = llvm::StringRef(raw).trim();
    return !(value == "0" || value.equals_insensitive("false") ||
             value.equals_insensitive("off"));
  }();
  return enabled;
}

/// Appends upstream MLIR's convert-elementwise-to-linalg pass, unless disabled
/// by `SPYRE_LINALG_COMPUTE`.
///
/// The pass rewrites every op carrying the `ElementwiseMappable` trait — all of
/// `arith` and `math` — that has at least one ranked-tensor operand into a
/// `linalg.generic` with identity indexing maps and the original scalar op in
/// its body. Ops with only scalar operands are left alone, which is required:
/// `linalg.reduce` combiner regions, index arithmetic, and `ktdp` address
/// computation all hold scalar `arith`.
///
/// `arith.constant` with a `DenseElementsAttr` result is also left alone; it
/// does not carry the trait. Converting those to `tensor.empty` + `linalg.fill`
/// is deferred — see `docs/designs/linalg-compute-normalization.md`.
void addElementwiseToLinalg(mlir::PassManager &pm) {
  if (elementwiseToLinalgEnabled())
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
}

} // namespace

void init_triton_spyre_passes_ttir_to_ktdp(py::module &&m) {
  // Pipeline: LowerDescriptorMemory → LowerScalarLoad → LowerComputeOps →
  //           ConvertElementwiseToLinalg → LowerInterTile → ConvertFunctions.
  // ConvertFunctions runs last because it replaces !tt.ptr args with index;
  // memory passes must consume !tt.ptr via getBasePtrAsIndex/ptrToIndex first.
  // LowerInterTile runs after LowerComputeOps (partials are linalg/tensor)
  // and before ConvertFunctions (reads work-slice function attributes that
  // ConvertFunctions would rewrite).
  //
  // ConvertElementwiseToLinalg runs after LowerComputeOps, not before: the
  // tt.reduce conversion builds linalg.reduce combiner regions out of scalar
  // arith, and running the elementwise conversion first would have nothing to
  // do for them anyway (they are scalar). Running it here also means the ops
  // LowerInterTile sees as reduction partials are already normalized.
  m.def("add_convert_ttir_to_ktdp", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerDescriptorMemoryPass());
    pm.addPass(mlir::triton::ktdp::createLowerScalarLoadPass());
    pm.addPass(mlir::triton::ktdp::createLowerComputeOpsPass());
    addElementwiseToLinalg(pm);
    pm.addPass(mlir::triton::ktdp::createLowerInterTilePass());
    pm.addPass(mlir::triton::ktdp::createConvertFunctionsPass());
  });
  // Individual pass bindings for debugging and testing.
  m.def("add_lower_inter_tile", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerInterTilePass());
  });
  m.def("add_lower_descriptor_memory", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerDescriptorMemoryPass());
  });
  m.def("add_lower_scalar_load", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerScalarLoadPass());
  });
  m.def("add_lower_compute_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerComputeOpsPass());
  });
  // Upstream MLIR's elementwise-to-linalg conversion, exposed separately so
  // tests can run it in isolation on hand-written MLIR. Unconditional: the
  // SPYRE_LINALG_COMPUTE switch gates the production pipeline, and a test that
  // asks for this pass by name should get it.
  m.def("add_convert_elementwise_to_linalg", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  });
  m.def("add_convert_functions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createConvertFunctionsPass());
  });
  m.def("add_distribute_work",
        [](mlir::PassManager &pm, const std::vector<int64_t> &grid) {
          pm.addPass(
              mlir::triton::ktdp::createDistributeWorkPass(grid));
        });
}

void init_triton_spyre_ir_utils(py::module &&m) {
  // get_integer_set_attr / get_affine_map_attr: return the printed form of
  // IntegerSetAttr / AffineMapAttr attributes on an operation.
  // The typed getters on ir.operation (get_str_attr, get_int_attr, …) do not
  // cover these MLIR attribute kinds, so we expose them here.
  m.def("get_integer_set_attr",
        [](mlir::Operation &self, const std::string &name) -> py::object {
          auto ret = self.getAttrOfType<mlir::IntegerSetAttr>(name);
          if (!ret)
            return py::none();
          std::string str;
          llvm::raw_string_ostream os(str);
          ret.getValue().print(os);
          return py::str(str);
        });
  m.def("get_affine_map_attr",
        [](mlir::Operation &self, const std::string &name) -> py::object {
          auto ret = self.getAttrOfType<mlir::AffineMapAttr>(name);
          if (!ret)
            return py::none();
          std::string str;
          llvm::raw_string_ostream os(str);
          ret.getValue().print(os);
          return py::str(str);
        });

  // Introspect the type of a result value.  Returns a dict with keys that
  // depend on the type kind.  For any ShapedType (tensor, memref, …):
  //   {"type_str": "memref<1024xf16>", "shape": [1024], "elem_type": "f16"}
  // For non-shaped types (index, i32, …):
  //   {"type_str": "index"}
  // Returns None if idx is out of range.
  m.def("get_result_info",
        [](mlir::Operation &self, unsigned idx) -> py::object {
          if (idx >= self.getNumResults())
            return py::none();
          auto type = self.getResult(idx).getType();
          py::dict d;
          std::string typeStr;
          llvm::raw_string_ostream typeOs(typeStr);
          type.print(typeOs);
          d["type_str"] = typeStr;
          if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
            auto shape = shaped.getShape();
            d["shape"] = std::vector<int64_t>(shape.begin(), shape.end());
            std::string elemStr;
            llvm::raw_string_ostream elemOs(elemStr);
            shaped.getElementType().print(elemOs);
            d["elem_type"] = elemStr;
          }
          return d;
        });
}

void init_triton_spyre(py::module &&m) {
  // Passes submodule
  auto passes = m.def_submodule("passes");
  init_triton_spyre_passes_ttir_to_ktdp(
      passes.def_submodule("ttir_to_ktdp"));

  // IR utilities submodule
  init_triton_spyre_ir_utils(m.def_submodule("ir_utils"));

  // Dialect registration
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::ktdp::KtdpDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::math::MathDialect>();
    context.appendDialectRegistry(registry);
  });
}
