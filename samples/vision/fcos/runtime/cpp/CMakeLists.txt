cmake_minimum_required(VERSION 3.15)
project(fcos_deploy VERSION 0.0.1)

include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(fcos-det ${sources})

target_link_libraries(fcos-det ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(fcos-det)

target_link_options(fcos-det PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS fcos-det DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/app)

