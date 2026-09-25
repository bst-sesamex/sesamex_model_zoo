# Common CMake setup for classification C++ deploy (bsnnrt 0.13.0).
# Links system libbsnnrt / libbsrpc under /usr/lib and uses bsnn.h from
# /usr/include/bsnn, with a vendored fallback in this project's include/.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    set(OpenCV_DIR "/opt/bstos/linux-23/sysroots/aarch64-bst-linux/usr/share/OpenCV" CACHE PATH "Fallback OpenCV directory")
    find_package(OpenCV REQUIRED)
endif()

set(DEVICE "" CACHE STRING "Target device. One of: A1000B, C1200, A2000")
set_property(CACHE DEVICE PROPERTY STRINGS A1000B C1200 A2000)

# Prefer the board-installed runtime (libbsnnrt.so / libbsrpc.so).
find_library(BSNN_LIB NAMES bsnnrt HINTS /usr/lib /usr/lib64 /lib NO_DEFAULT_PATH)
if(NOT BSNN_LIB)
    find_library(BSNN_LIB NAMES bsnnrt)
endif()
if(NOT BSNN_LIB)
    message(FATAL_ERROR "libbsnnrt.so not found. Install bsnnrt 0.13.0 into /usr/lib.")
endif()

find_library(BSRPC_LIB NAMES bsrpc HINTS /usr/lib /usr/lib64 /lib NO_DEFAULT_PATH)
if(NOT BSRPC_LIB)
    find_library(BSRPC_LIB NAMES bsrpc)
endif()
if(NOT BSRPC_LIB)
    message(FATAL_ERROR "libbsrpc.so not found. Install bsnnrt 0.13.0 into /usr/lib.")
endif()

message(STATUS "DEVICE=${DEVICE}")
message(STATUS "BSNN_LIB=${BSNN_LIB}")
message(STATUS "BSRPC_LIB=${BSRPC_LIB}")

set(_BSNN_HEADER_DIRS
    /usr/include/bsnn
    /usr/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/py_utils/deploy
    /userdata/new_bsnn/include
)
include_directories(
    ${_BSNN_HEADER_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Collect sources from src directory
if(DEVICE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src-${DEVICE}")
    message(STATUS "Using source directory: src-${DEVICE}")
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src-${DEVICE} sources)
else()
    message(STATUS "Using source directory: src")
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src sources)
endif()

# OpenCV on aarch64 sysroot may pull Armadillo; link BLAS/LAPACK explicitly.
set(OPENCV_SYSROOT_MATH_LIBS "")
set(OPENCV_SYSROOT_MATH_ROOT "")
set(_MATH_SYSROOT "")
if(CMAKE_SYSROOT)
    set(_MATH_SYSROOT "${CMAKE_SYSROOT}")
elseif(CMAKE_CXX_COMPILER_ARG1 MATCHES ".*--sysroot=([^ ]+)")
    set(_MATH_SYSROOT "${CMAKE_MATCH_1}")
endif()
set(_BLAS_SO "${_MATH_SYSROOT}/usr/lib/aarch64-linux-gnu/blas/libblas.so.3")
set(_LAPACK_SO "${_MATH_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3")
set(_GFORTRAN_SO "${_MATH_SYSROOT}/usr/lib/aarch64-linux-gnu/libgfortran.so.5")
if(_MATH_SYSROOT AND EXISTS "${_BLAS_SO}" AND EXISTS "${_LAPACK_SO}" AND EXISTS "${_GFORTRAN_SO}")
    set(OPENCV_SYSROOT_MATH_ROOT "${_MATH_SYSROOT}")
    set(OPENCV_SYSROOT_MATH_LIBS "${_BLAS_SO}" "${_LAPACK_SO}" "${_GFORTRAN_SO}")
endif()

function(target_link_opencv_sysroot_math _tgt)
    if(OPENCV_SYSROOT_MATH_LIBS)
        target_link_options(${_tgt} PRIVATE
            "LINKER:-rpath-link,${OPENCV_SYSROOT_MATH_ROOT}/usr/lib/aarch64-linux-gnu"
            "LINKER:-rpath-link,${OPENCV_SYSROOT_MATH_ROOT}/usr/lib/aarch64-linux-gnu/blas"
            "LINKER:-rpath-link,${OPENCV_SYSROOT_MATH_ROOT}/usr/lib/aarch64-linux-gnu/lapack"
        )
        target_link_libraries(${_tgt} ${OPENCV_SYSROOT_MATH_LIBS})
    endif()
endfunction()
