# External KTDP dialect definitions from ktir-mlir-frontend.
set(KTIR_MLIR_FRONTEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ktir-mlir-frontend")
if(NOT EXISTS "${KTIR_MLIR_FRONTEND_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "ktir-mlir-frontend not found at ${KTIR_MLIR_FRONTEND_DIR}. "
    "Run: git submodule update --init")
endif()
add_subdirectory(${KTIR_MLIR_FRONTEND_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ktir-mlir-frontend EXCLUDE_FROM_ALL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
# C++ tool binaries (spyre-triton-opt).  The MLIR convention is tools/ but
# Triton reserves third_party/<backend>/tools/ for Python tool packages
# installed into triton.tools.extra (see setup.py BackendInstaller).  We use
# bin/ to follow Triton's top-level bin/ pattern for C++ opt binaries.
add_subdirectory(bin)
add_subdirectory(test)

if(TRITON_BUILD_PYTHON_MODULE)
  # MLIRLinalgTransforms is for this file's own
  # createConvertElementwiseToLinalgPass, which the plugin exposes as
  # add_convert_elementwise_to_linalg.  
  # The three pass libraries stay named: unlike spyre-triton-opt, this file
  # reaches individual create*Pass factories, not just the registration entry
  # point.
  #
  # TTSIR is for the one op builder this file exposes to the frontend. Reached
  # transitively through SpyreRegisterEverything, but named because
  # triton_spyre.cc includes the dialect's header directly -- and DEPENDS on
  # TTSIncGen for the same reason, so a clean parallel build does not rely on the
  # link edge to order that tablegen ahead of this file.
  add_triton_plugin(TritonSpyre
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_spyre.cc

    DEPENDS
    TTSIncGen

    LINK_LIBS SpyreRegisterEverything
              SpyrePipeline
              TritonToKTIR KTDPTransforms SpyreTransforms
              TTSIR
              # getDescriptorLogicalLayout, for the descriptor-layout query.
              SpyreUtils
              MLIRLinalgTransforms)
  target_link_libraries(TritonSpyre PRIVATE Python3::Module pybind11::headers)
endif()
