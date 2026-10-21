cmake_minimum_required(VERSION 3.15)
project(demo VERSION 0.0.1)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/app)

include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(lightstereo ${sources})

target_link_libraries(lightstereo ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(lightstereo)

target_link_options(lightstereo PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS lightstereo DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/app)
