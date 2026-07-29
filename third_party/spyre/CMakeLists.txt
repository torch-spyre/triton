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
  add_triton_plugin(TritonSpyre
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_spyre.cc
    LINK_LIBS KTDPTransforms KTIR::KtdpDialect)
  target_link_libraries(TritonSpyre PRIVATE Python3::Module pybind11::headers)
endif()
