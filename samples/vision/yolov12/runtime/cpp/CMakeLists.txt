cmake_minimum_required(VERSION 3.15)
project(demo VERSION 0.0.1)

# Use modern CMake to set language standard
# Common setup
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

# Additional project-specific includes/links can be added here
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include)
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bst_isp/lib)

add_executable(yolo-det ${sources})

target_link_libraries(yolo-det ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(yolo-det)

# Optionally strip the binary at link time (matches previous -s intent)
target_link_options(yolo-det PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS yolo-det DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/app)
