cmake_minimum_required(VERSION 3.15)
project(unet_deploy VERSION 0.0.1)

# Common setup
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(unet-seg ${sources})

target_link_libraries(unet-seg ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(unet-seg)

target_link_options(unet-seg PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS unet-seg DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../prebuild/app)


