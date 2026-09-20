//===- triton_spyre.cc - Pybind11 bindings for Spyre backend --------------===//
//
// Exposes the KTDP dialect and Spyre lowering passes to Python via pybind11.
// This is compiled as part of the main libtriton shared library.
// The entry point init_triton_spyre() is called from main.cc via the
// FOR_EACH_P(INIT_BACKEND, ...) macro.
//
//===----------------------------------------------------------------------===//

#include "RegisterEverything.h"
// The two stage pipelines this file exposes. No pass headers: nothing here names
// an individual pass any more, so the create* factories are reached only from
// Pipeline.cpp and from each group's own registration.
#include "Pipeline.h"
// The one dialect of ours a kernel is authored in, for the op builder below.
#include "Dialect/TTS/IR/Dialect.h"
// TritonOpBuilder, defined header-only under python/src/. That directory is on
// the include path here because the top-level CMakeLists adds it before it adds
// third_party/<backend>, so a backend can reach it without naming a path.
#include "ir.h"
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
  // One entry point per compile stage, and nothing finer. The pass lists are in
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
      [](mlir::PassManager &pm, const std::vector<int64_t> &grid) {
        mlir::triton::spyre::TTIRToKTIRPipelineOptions options;
        options.grid = grid;
        mlir::triton::spyre::buildTTIRToKTIRPipeline(pm, options);
      },
      py::arg("pm"), py::arg("grid") = std::vector<int64_t>{});
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
  // No per-pass bindings. There were twelve, and by the end their only caller was
  // the table that turned a `SpyreOptions.required_fixes` pass *name* into a
  // pass; with that option gone, a stage's pass list is chosen in C++ from typed
  // factories, where a name cannot be misspelled. A single pass is still
  // drivable, from `spyre-triton-opt` by its registered CLI flag, which is where
  // the .mlir tests reach it.
}

void init_triton_spyre_ir_builders(py::module &&m) {
  // Op builders for the `tts` dialect, called from the Triton frontend --
  // tl.spyre_tensor_layout, through triton.language.semantic.
  //
  // The frontend reaches this as `from triton._C.libtriton import spyre`, lazily
  // -- an import at module scope in semantic.py would make every backend's
  // frontend depend on a submodule only a Spyre build has.
  //
  // TritonOpBuilder crosses from ir.cc's bindings into this function's signature
  // even though it is bound `py::module_local()`, because module-local means the
  // pybind *module*, and this file is compiled into the same libtriton as ir.cc
  // -- one PYBIND11_MODULE, this a submodule of it. `pass_manager` above is the
  // same arrangement and has always worked.
  m.def("create_tensor_layout",
        [](TritonOpBuilder &self, mlir::Value &desc,
           std::vector<int64_t> &physSrc, std::vector<int64_t> &physOp,
           std::vector<int64_t> &physArg) -> void {
          // LOAD, not register. `spyre.load_dialects` has already put tts in the
          // context's registry -- but a registry entry only makes a dialect
          // *loadable*, and constructing an op needs it *loaded*. Every other
          // producer of one of our ops is a pass, where the pass manager loads
          // `dependentDialects` for us; the frontend has no pass manager, so this
          // is the one place that has to ask. Without it the failure is
          // `LLVM ERROR: tts.tensor_layout created with unregistered dialect`,
          // an abort with no Python traceback to the kernel line. Idempotent, so
          // per-op is the right granularity: it costs a map lookup and there is
          // no earlier hook in this module that the frontend is guaranteed to
          // cross.
          self.getContext()->loadDialect<mlir::triton::tts::TTSDialect>();

          auto &builder = self.getBuilder();
          self.create<mlir::triton::tts::TensorLayoutOp>(
              desc, builder.getDenseI64ArrayAttr(physSrc),
              builder.getDenseI64ArrayAttr(physOp),
              builder.getDenseI64ArrayAttr(physArg));
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

  // Op builders the frontend calls, and IR introspection the tests call. Two
  // submodules because the first writes IR and the second only reads it.
  init_triton_spyre_ir_builders(m.def_submodule("ir_builders"));
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
