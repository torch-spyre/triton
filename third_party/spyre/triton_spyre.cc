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
// The one dialect of ours a kernel is authored in, for the op builder below and
// for the coordinate-map evaluator the descriptor-layout query reads.
#include "Dialect/TTS/IR/Dialect.h"
// getDescriptorLogicalLayout, shared with LowerDescriptorMemory so the footprint
// this file reports is computed from the extents that pass builds the view with.
#include "Utils/Utility.h"
// triton::isKernel, for finding the entry function the way ir.cc's
// getTensorDescMetadata does.
#include "triton/Dialect/Triton/IR/Utility.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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
  // tl.spyre_tensor_layout and tl.spyre_pin, through
  // triton.language.semantic.
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

  // tl.spyre_pin. The memory space arrives as the string the kernel wrote and is
  // stored as one: `tts.pin` spells it that way so the dialect keeps defining no
  // attribute type, and its verifier is what checks the string against ktdp's
  // enum. The frontend restates the two names as well, so that a misspelling is a
  // traceback at the pin rather than a verifier failure after the whole function
  // has been traced -- but neither place symbolizes it, and the op is where the
  // rule lives.
  //
  // `address` is optional, and a missing one is a null Value rather than a
  // sentinel: that is how ODS spells an absent optional operand, and it keeps
  // "the author stated no address" distinguishable from "the author stated 0".
  m.def(
      "create_pin",
      [](TritonOpBuilder &self, mlir::Value &value,
         const std::string &memorySpace,
         std::optional<mlir::Value> address) -> void {
        // LOAD, not register -- see create_tensor_layout above for why the
        // frontend is the one place that has to ask.
        self.getContext()->loadDialect<mlir::triton::tts::TTSDialect>();

        auto &builder = self.getBuilder();
        self.create<mlir::triton::tts::PinOp>(
            value, address ? *address : mlir::Value(),
            builder.getStringAttr(memorySpace));
      },
      py::arg("builder"), py::arg("value"), py::arg("memory_space"),
      py::arg("address") = py::none());
}

/// One `tts.tensor_layout` marker, reduced to what a footprint is computed from.
///
/// `ptrIndex` is the ORDINAL of the entry function's pointer argument the
/// descriptor is based on -- the i-th `!tt.ptr`, not the i-th argument. That is
/// the key the whole launch ABI already uses: `_segment_addresses` hands segment
/// i to pointer i, `_address_args` collects the `*`-typed signature entries in
/// the same order, and the correction flit is walked positionally. A parameter
/// *name* would be a better key and is not available: Triton records no argument
/// names in the IR, and a compile stage is handed `(module, metadata)` and never
/// the source, so the name exists only at launch, where the launcher has the
/// signature and resolves it for the diagnostic.
struct MarkerFootprint {
  int64_t ptrIndex;
  bool isLoad = false;
  bool isStore = false;
  /// False when a physical extent is not a compile-time answer, in which case
  /// both arrays are empty and the entry carries a null footprint.
  bool staticFootprint = false;
  /// The finished torch-spyre pair, from `tts::evaluateDeviceLayout` -- extents,
  /// stride rule and unit-axis padding already applied. Nothing on the Python
  /// side derives anything from these, which is why the coordinate map itself is
  /// not reported: a caller that could not evaluate it has no use for it, and a
  /// caller that could would be a second evaluator.
  std::vector<int64_t> deviceSize, strideMap;
};

/// What the kernel does with the buffer, as the metadata entry's `access`.
///
/// An annotated descriptor nothing reads or writes is reported as `"none"`
/// rather than dropped: the buffer still has to be big enough if the kernel is
/// later edited to use it, and "no access" is a more useful thing for a
/// diagnostic to say than a missing entry.
static const char *accessName(const MarkerFootprint &fp) {
  if (fp.isLoad && fp.isStore)
    return "load_store";
  if (fp.isStore)
    return "store";
  if (fp.isLoad)
    return "load";
  return "none";
}

void init_triton_spyre_ir_utils(py::module &&m) {
  // get_descriptor_layouts: how much device memory each annotated buffer of this
  // kernel actually needs.
  //
  // A layout annotation can ask for a device buffer LARGER than its host tensor --
  // a splat replicates a statistic across a stick -- and nothing downstream can
  // work that out: the allocator sees only the host shape and the launcher only
  // device residency, so the overrun is silent until it corrupts a neighbour. This
  // query is where the compiler writes its number down. It IS
  // metadata["device_layouts"] -- `_make_ktir` stores the list as it comes back,
  // the launcher bounds-checks a tensor against it, and the fixture harness
  // allocates from it. One entry per annotated buffer, of JSON-survivable values:
  //
  //     {"ptr_index": 1, "device_size": [1, 64, 64], "stride_map": [-1, 1, -1],
  //      "access": "store"}
  //
  // `device_size` and `stride_map` are None TOGETHER when any physical extent is
  // not known at compile time -- a descriptor taking its shape from a runtime
  // i32 argument, which `matmul__spyre_stick_parallel_dynamic` does. Such a
  // kernel is unlaunchable today anyway; the point of recording the entry at all
  // is that its absence and its emptiness mean different things. A MISSING entry
  // for a pointer argument is not a fault: it means the kernel made no claim
  // about that buffer, which is every unannotated descriptor.
  //
  // One query rather than a walk in Python, and the finished pair rather than
  // the coordinate map, for the same reason: the derivation is
  // `tts::evaluateDeviceLayout`, over the extents `tts::applyCoordMap` gives,
  // which is what `rewrite-descriptor-layout-generic` builds the physical memref
  // with -- so nothing on the Python side has to know the coordinate-op
  // numbering, and no transcription of the arithmetic can drift from it. It had
  // one, and the numbering with it. The precedent for the query shape is
  // `getTensorDescMetadata` in python/src/ir.cc, which likewise walks a kernel's
  // arguments and hands back dicts.
  //
  // Must be called while the entry point is still a `tt.func` taking `!tt.ptr`
  // arguments, and while the marker ops are still present -- so before the
  // TTIR→KTIR pipeline, alongside metadata["name"] and the base-address
  // inference. `LowerTTSMarkers` turns each marker into an attribute and
  // `ConvertFunctions` retypes the pointers; after either, this returns nothing
  // and says nothing about why.
  m.def("get_descriptor_layouts", [](mlir::ModuleOp &mod) -> py::list {
    using namespace mlir;

    py::list result;

    triton::FuncOp kernel;
    mod.walk([&](triton::FuncOp func) {
      if (!triton::isKernel(func))
        return WalkResult::skip();
      kernel = func;
      return WalkResult::interrupt();
    });
    if (!kernel)
      return result;

    // Argument index -> pointer ordinal, for the arguments that are pointers.
    llvm::DenseMap<unsigned, int64_t> ptrOrdinal;
    int64_t seen = 0;
    for (auto [i, ty] : llvm::enumerate(kernel.getFunctionType().getInputs()))
      if (isa<triton::PointerType>(ty))
        ptrOrdinal[i] = seen++;

    // Collected keyed by pointer ordinal so that a second marker on the same
    // argument is visible as one. Two footprints over one buffer describe a
    // union this metadata cannot spell -- there is one `device_size` per entry --
    // and the honest answer is to make no claim rather than a claim that happens
    // to be the larger of two. Recorded as a value that fails the check below.
    llvm::MapVector<int64_t, MarkerFootprint> byPtr;
    llvm::DenseSet<int64_t> ambiguous;

    mod.walk([&](triton::tts::TensorLayoutOp marker) {
      auto descOp = marker.getDesc().getDefiningOp<triton::MakeTensorDescOp>();
      if (!descOp)
        return;
      // The base must be a pointer argument of the entry function itself. A
      // descriptor over a computed pointer has a footprint that is a function of
      // that arithmetic, not of this layout, so there is nothing here to claim.
      auto base = dyn_cast<BlockArgument>(descOp.getBase());
      if (!base || base.getOwner() != &kernel.getBody().front())
        return;
      auto it = ptrOrdinal.find(base.getArgNumber());
      if (it == ptrOrdinal.end())
        return;

      MarkerFootprint fp;
      fp.ptrIndex = it->second;

      SmallVector<int64_t> sizes, strides;
      triton::spyre::getDescriptorLogicalLayout(descOp, sizes, strides);

      ArrayRef<int64_t> src = marker.getPhysSrc();
      ArrayRef<int64_t> op = marker.getPhysOp();
      ArrayRef<int64_t> arg = marker.getPhysArg();

      // Guarded rather than assumed: this runs on whatever a kernel authored,
      // and the evaluator indexes the logical arrays with phys_src[k]. The op's
      // own verifier has already checked the bound against the descriptor's
      // BLOCK rank, which equals the tensor rank -- but a module handed in as
      // text has not necessarily been verified.
      bool inRange = src.size() == op.size() && src.size() == arg.size();
      for (int64_t d : src)
        inRange &= d >= 0 && d < (int64_t)sizes.size();
      if (inRange) {
        SmallVector<int64_t> deviceSize, strideMap;
        if (triton::tts::evaluateDeviceLayout(sizes, strides, src, op, arg,
                                              deviceSize, strideMap)) {
          fp.staticFootprint = true;
          fp.deviceSize.assign(deviceSize.begin(), deviceSize.end());
          fp.strideMap.assign(strideMap.begin(), strideMap.end());
        }
      }

      for (Operation *user : descOp.getResult().getUsers()) {
        if (isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(user))
          fp.isLoad = true;
        else if (isa<triton::DescriptorStoreOp, triton::DescriptorScatterOp>(
                     user))
          fp.isStore = true;
      }

      if (!byPtr.insert({fp.ptrIndex, fp}).second)
        ambiguous.insert(fp.ptrIndex);
    });

    for (const auto &entry : byPtr) {
      if (ambiguous.contains(entry.first))
        continue;
      const MarkerFootprint &fp = entry.second;
      py::dict d;
      d["ptr_index"] = fp.ptrIndex;
      // None, not an empty list: a dynamic extent means there is no claim, and
      // an empty list would read as a rank-0 one. The two go None TOGETHER --
      // a stride map without extents describes nothing.
      d["device_size"] = fp.staticFootprint
                             ? py::cast(fp.deviceSize)
                             : py::cast<py::object>(py::none());
      d["stride_map"] = fp.staticFootprint
                            ? py::cast(fp.strideMap)
                            : py::cast<py::object>(py::none());
      d["access"] = accessName(fp);
      result.append(d);
    }
    return result;
  });

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
