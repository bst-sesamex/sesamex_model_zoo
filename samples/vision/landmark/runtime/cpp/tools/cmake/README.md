# Common CMake Modules (placeholder)

This folder is reserved for shared CMake modules to be vendored into packaged projects.

Planned usage (future):
- Place shared modules here (e.g. BSNN runtime resolution, common OpenCV lookup).
- Packaging flow copies the entire `tools/` directory into `<project>/Deploy/code/tools/`.
- Project `Deploy/code/CMakeLists.txt` can then do:

  - `include(${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/<module>.cmake)`
  - or append to `CMAKE_MODULE_PATH`.
