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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_spyre_passes_ttir_to_ktdp(py::module &&m) {
  // Pipeline: LowerDescriptorMemory → LowerComputeOps →
  // RewriteDescriptorLayout → ConvertFunctions.
  //
  // RewriteDescriptorLayout runs LAST (before ConvertFunctions) so that
  // tt.dot is already linalg.matmul before operands are physicalized — tt.dot
  // requires 2-D operands and would mis-lower a rank-3 physical operand.
  // The marker (tt.spyre_tensor_layout) persists through the two intervening
  // passes: its desc operand auto-re-points at the UnrealizedConversionCast
  // bridge left by LowerDescriptorMemory Walk 1, and both passes mark the
  // marker legal so it is never flagged as unconverted.
  //
  // ConvertFunctions runs last because it replaces !tt.ptr args with index;
  // memory passes must consume !tt.ptr via getBasePtrAsIndex first.
  m.def("add_convert_ttir_to_ktdp", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerDescriptorMemoryPass());
    pm.addPass(mlir::triton::ktdp::createLowerComputeOpsPass());
    pm.addPass(mlir::triton::ktdp::createRewriteDescriptorLayoutPass());
    pm.addPass(mlir::triton::ktdp::createConvertFunctionsPass());
  });
  // Individual pass bindings for debugging and testing.
  m.def("add_rewrite_descriptor_layout", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createRewriteDescriptorLayoutPass());
  });
  m.def("add_lower_descriptor_memory", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerDescriptorMemoryPass());
  });
  m.def("add_lower_compute_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createLowerComputeOpsPass());
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
