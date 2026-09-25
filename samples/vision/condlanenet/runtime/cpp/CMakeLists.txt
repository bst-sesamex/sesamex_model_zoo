cmake_minimum_required(VERSION 3.15)
project(condlanenet_deploy VERSION 0.0.1)

include("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/common_setup.cmake")

add_executable(condlanenet ${sources})

target_link_libraries(condlanenet ${OpenCV_LIBS} ${BSNN_LIB} ${BSRPC_LIB} pthread stdc++fs dl)
target_link_opencv_sysroot_math(condlanenet)

target_link_options(condlanenet PRIVATE $<$<CONFIG:Release>:-s>)
install(TARGETS condlanenet DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../prebuild/app)
