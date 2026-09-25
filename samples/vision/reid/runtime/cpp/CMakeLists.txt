cmake_minimum_required(VERSION 3.15)
project(reid_deploy VERSION 0.0.1)

# Common setup
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(reid-pair-match ${sources})

target_link_libraries(reid-pair-match ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(reid-pair-match)

target_link_options(reid-pair-match PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS reid-pair-match DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/app)


