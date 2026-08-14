//===- triton_spyre.cc - Pybind11 bindings for Spyre backend --------------===//
//
// Exposes the KTDP dialect and Spyre lowering passes to Python via pybind11.
// This is compiled as part of the main libtriton shared library.
// The entry point init_triton_spyre() is called from main.cc via the
// FOR_EACH_P(INIT_BACKEND, ...) macro.
//
//===----------------------------------------------------------------------===//

#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_spyre_passes_ttir_to_ktdp(py::module &&m) {
  // Pass order built by add_convert_ttir_to_ktdp:
  //
  //     LowerDescriptorMemory      [LowerPointerChainMemory — planned,
  //              │                  not yet implemented; would handle the
  //              │                  tensor-of-pointers tt.load that
  //              │                  LowerScalarLoad leaves legal]
  //              ↓
  //       LowerScalarLoad
  //              ↓
  //       LowerComputeOps
  //              ↓
  //   RewriteDescriptorLayout      [runs after LowerComputeOps so tt.dot is
  //              │                  already linalg.matmul before its operands
  //              │                  are physicalized]
  //              ↓
  //        LowerInterTile
  //              ↓
  //       ConvertFunctions
  //
  // This is only the nested pipeline. DistributeWork and canonicalize + CSE
  // run after it, added separately by the `ktir` stage in
  // third_party/spyre/backend/compiler.py.
  //
  // Ordering constraints (each pass also states its own in Passes.td):
  // ConvertFunctions runs last because it replaces !tt.ptr args with index;
  // memory passes must consume !tt.ptr via getBasePtrAsIndex/ptrToIndex first.
  // LowerInterTile runs after LowerComputeOps (partials are linalg/tensor)
  // and before ConvertFunctions (reads work-slice function attributes that
  // ConvertFunctions would rewrite).
  m.def(
      "add_convert_ttir_to_ktdp",
      [](mlir::PassManager &pm, const std::string &data_layout) {
        pm.addPass(mlir::triton::ktdp::createLowerDescriptorMemoryPass());
        pm.addPass(mlir::triton::ktdp::createLowerScalarLoadPass());
        pm.addPass(mlir::triton::ktdp::createLowerComputeOpsPass());
        pm.addPass(mlir::triton::ktdp::createRewriteDescriptorLayout(
            mlir::triton::ktdp::RewriteDescriptorLayoutOptions{data_layout}));
        pm.addPass(mlir::triton::ktdp::createLowerInterTilePass());
        pm.addPass(mlir::triton::ktdp::createConvertFunctionsPass());
      },
      py::arg("pm"), py::arg("data_layout") = "device");
  // Individual pass bindings. add_convert_ttir_to_ktdp above is the default
  // order, but a caller that needs a different one — a subset of the passes,
  // a repeat, or an extra pass slotted between two of them — builds the
  // sequence from these instead. Used by the `required_fixes` mechanism in
  // third_party/spyre/backend/compiler.py to insert correctness patches at a
  // chosen point in the pipeline, and by the per-pass unit tests that run one
  // pass over inline MLIR. Every pass in the default order has a binding here,
  // so any reordering expressible in C++ is also expressible from Python.
  //
  m.def("add_convert_elementwise_to_linalg", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  });
  m.def(
      "add_rewrite_descriptor_layout",
      [](mlir::PassManager &pm, const std::string &data_layout) {
        pm.addPass(mlir::triton::ktdp::createRewriteDescriptorLayout(
            mlir::triton::ktdp::RewriteDescriptorLayoutOptions{data_layout}));
      },
      py::arg("pm"), py::arg("data_layout") = "device");
  // Not in add_convert_ttir_to_ktdp above: this is a fix pass, spliced into the
  // pipeline from Python via SpyreOptions.required_fixes. It must be anchored on
  // convert_elementwise_to_linalg, which is the pass that creates the ins/outs
  // aliasing it removes; anchoring it on anything earlier is a silent no-op,
  // since the pass only rewrites aliasing that already exists.
  m.def("add_unalias_linalg_outs", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createUnaliasLinalgOutsPass());
  });
  // Also a fix pass, and also anchored on the pass that creates what it removes:
  // lower_compute_ops is what gives every tt.reduce a linalg.fill init. The
  // scheduler's allowlist has no linalg.fill, so without this the KTIR is
  // rejected at pass 00; see the pass description for why the gate is zero
  // rather than the combiner's neutral element.
  m.def("add_drop_reduction_init_fill", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createDropReductionInitFillPass());
  });
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
  m.def("add_convert_functions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ktdp::createConvertFunctionsPass());
  });
  m.def(
      "add_distribute_work",
      [](mlir::PassManager &pm, const std::vector<int64_t> &grid) {
        pm.addPass(mlir::triton::ktdp::createDistributeWorkPass(grid));
      },
      py::arg("pm"), py::arg("grid"));
  // Opt-in only: MaterializeBaseAddresses is deliberately absent from
  // add_convert_ttir_to_ktdp above. It changes the kernel's calling
  // convention (base-address arguments become arith.constant and leave the
  // signature), which only the dataflow-scheduler path wants; the default
  // argument-passing path must stay byte-identical. Reached via
  // required_fixes = {"materialize_base_addresses": "convert_functions"}.
  m.def(
      "add_materialize_base_addresses",
      [](mlir::PassManager &pm, const std::vector<int64_t> &base_addresses) {
        pm.addPass(mlir::triton::ktdp::createMaterializeBaseAddressesPass(
            base_addresses));
      },
      py::arg("pm"), py::arg("base_addresses"));
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
