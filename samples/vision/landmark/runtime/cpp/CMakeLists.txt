cmake_minimum_required(VERSION 3.15)
project(landmark_deploy VERSION 0.0.1)

# Common setup
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(landmark-infer ${sources})

target_link_libraries(landmark-infer ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(landmark-infer)

target_link_options(landmark-infer PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS landmark-infer DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/app)


