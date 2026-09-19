//===- triton_spyre.cc - Pybind11 bindings for Spyre backend --------------===//
//
// Exposes the KTDP dialect and Spyre lowering passes to Python via pybind11.
// This is compiled as part of the main libtriton shared library.
// The entry point init_triton_spyre() is called from main.cc via the
// FOR_EACH_P(INIT_BACKEND, ...) macro.
//
//===----------------------------------------------------------------------===//

#include "RegisterEverything.h"
// The two stage pipelines this file exposes.
#include "Pipeline.h"
// All three pass groups: this file reaches create* entry points from each --
// the conversions and the top-level transforms by their hand-declared
// factories, RewriteDescriptorLayout through the options struct tablegen
// generates into the KTDP transforms header. Only the individual pass bindings
// need these now; the pipelines come from Pipeline.h.
#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_spyre_passes_ttir_to_ktdp(py::module &&m) {
  // One entry point per compile stage. The pass lists are in
  // third_party/spyre/lib/Pipeline.cpp, spelled once: this file used to carry a
  // fused `add_convert_ttir_to_ktdp` alongside a pass-by-pass loop in
  // backend/compiler.py, and the two drifted -- the fused one never learned
  // about the two passes the Python loop always spliced in, and no test could
  // see it, because every .mlir test drives a single pass. The same lists are
  // reachable as `spyre-triton-opt --spyre-ttir-to-ktir` /
  // `--spyre-prepare-spyrecode`, which is what lets a lit test cover a whole
  // stage.
  //
  // Options arrive as values rather than as an options string. Pipeline
  // registration parses a string, which suits `grid` and suits the base
  // addresses badly: they are element indices up to 2^35, positional, and the
  // empty list means "this kernel has no pointer arguments" rather than
  // "unset".
  m.def(
      "add_ttir_to_ktir_pipeline",
      [](mlir::PassManager &pm, const std::string &data_layout,
         const std::vector<int64_t> &grid,
         const std::function<void(const std::string &)> &fixes_at) {
        mlir::triton::spyre::TTIRToKTIRPipelineOptions options;
        options.dataLayout = data_layout;
        options.grid = grid;
        // `fixes_at` appends to this same pass manager, from Python, at each
        // anchor the builder announces -- which is how SpyreOptions
        // .required_fixes lands a pass in the position its anchor names. The
        // pass manager is not handed back across the boundary: the caller
        // already holds it, so the callback takes only the anchor name. Both go
        // when required_fixes does.
        if (fixes_at)
          options.anchorHook = [&](llvm::StringRef anchor) {
            fixes_at(anchor.str());
          };
        mlir::triton::spyre::buildTTIRToKTIRPipeline(pm, options);
      },
      py::arg("pm"), py::arg("data_layout") = "device",
      py::arg("grid") = std::vector<int64_t>{},
      py::arg("fixes_at") = std::function<void(const std::string &)>{});
  m.def(
      "add_spyrecode_pipeline",
      [](mlir::PassManager &pm, bool bind_base_addresses,
         const std::vector<int64_t> &base_addresses) {
        mlir::triton::spyre::SpyrecodePipelineOptions options;
        options.bindBaseAddresses = bind_base_addresses;
        options.baseAddresses = base_addresses;
        mlir::triton::spyre::buildSpyrecodePipeline(pm, options);
      },
      py::arg("pm"), py::arg("bind_base_addresses") = false,
      py::arg("base_addresses") = std::vector<int64_t>{});
  // Individual pass bindings. They are no longer how the pipeline is built --
  // see the two entry points above -- and exist now for one reason only: they
  // are the table that turns a `SpyreOptions.required_fixes` pass *name* into a
  // pass. Keeping one is what lets a caller name it; removing one turns naming
  // it into a loud ValueError from _add_ktdp_pass. They go with that option.
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
  // Both of these are in buildTTIRToKTIRPipeline's own list now; the bindings
  // remain because a caller can still name either as a fix, and because they
  // are what the per-pass unit tests drive over inline MLIR.
  m.def("add_unalias_linalg_outs", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createUnaliasLinalgOutsPass());
  });
  m.def("add_drop_reduction_init_fill", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createDropReductionInitFillPass());
  });
  m.def("add_lower_inter_tile", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createLowerInterTilePass());
  });
  m.def("add_lower_descriptor_memory", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createLowerDescriptorMemoryPass());
  });
  m.def("add_lower_scalar_load", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createLowerScalarLoadPass());
  });
  m.def("add_lower_compute_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createLowerComputeOpsPass());
  });
  m.def("add_lower_spyre_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createLowerSpyreOpsPass());
  });
  m.def("add_convert_functions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spyre::createConvertFunctionsPass());
  });
  m.def(
      "add_distribute_work",
      [](mlir::PassManager &pm, const std::vector<int64_t> &grid) {
        pm.addPass(mlir::triton::spyre::createDistributeWorkPass(grid));
      },
      py::arg("pm"), py::arg("grid"));
  // MaterializeBaseAddresses is deliberately absent from the `ktir` stage's
  // list: it changes the kernel's calling convention, so the cached .ktir
  // artifact would lose its arguments. buildSpyrecodePipeline installs it, under
  // its own flag. This binding is for a caller that wants it during the `ktir`
  // stage instead, via
  // required_fixes = {"materialize_base_addresses": "convert_functions"}.
  m.def(
      "add_materialize_base_addresses",
      [](mlir::PassManager &pm, const std::vector<int64_t> &base_addresses) {
        pm.addPass(mlir::triton::spyre::createMaterializeBaseAddressesPass(
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

  // Dialect registration. Appends to a context Triton has already populated
  // (python/src/ir.cc load_dialects runs first), so this adds only what the
  // Spyre passes need on top.
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::triton::spyre::registerDialects(registry);
    context.appendDialectRegistry(registry);
  });
}
