cmake_minimum_required(VERSION 3.15)
project(yolov5_deploy VERSION 0.0.1)

include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(yolov5 ${sources})

target_link_libraries(yolov5 ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(yolov5)

target_link_options(yolov5 PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS yolov5 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../prebuild/app)
