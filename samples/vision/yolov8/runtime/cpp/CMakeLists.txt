cmake_minimum_required(VERSION 3.15)
project(demo VERSION 0.0.1)

# Use modern CMake to set language standard
# Common setup
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

# Additional project-specific includes/links can be added here
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include)
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bst_isp/lib)

add_executable(yolov8 ${sources})

target_link_libraries(yolov8 ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(yolov8)

# Optionally strip the binary at link time (matches previous -s intent)
target_link_options(yolov8 PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS yolov8 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../prebuild/app)
