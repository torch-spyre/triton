set(SPYRE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SPYRE_TEST_DEPENDS
  spyre-triton-opt
  FileCheck
)

add_lit_testsuite(check-spyre-triton "Running the Spyre+Triton regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${SPYRE_TEST_DEPENDS}
)
set_target_properties(check-spyre-triton PROPERTIES FOLDER "Tests")
