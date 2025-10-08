cmake_minimum_required(VERSION 3.15)
project(loop-cgal VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_library(_loop_cgal MODULE
    loop_cgal/bindings.cpp
    src/mesh.cpp
    src/meshutils.cpp
    src/globals.cpp
)

target_link_libraries(_loop_cgal PRIVATE pybind11::module CGAL::CGAL)
target_include_directories(_loop_cgal PRIVATE ${CMAKE_SOURCE_DIR}/src)

set_target_properties(_loop_cgal PROPERTIES PREFIX "" SUFFIX ".pyd")

# Windows: copy required DLLs from VCPKG_ROOT
if(WIN32)
    if(NOT DEFINED ENV{VCPKG_ROOT})
        message(FATAL_ERROR "VCPKG_ROOT environment variable is not set!")
    endif()
    set(VCPKG_ROOT $ENV{VCPKG_ROOT})
    message(STATUS "Using VCPKG_ROOT: ${VCPKG_ROOT}")

    add_custom_command(TARGET _loop_cgal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VCPKG_ROOT}/packages/gmp_x64-windows/bin/gmp-10.dll"
            "$<TARGET_FILE_DIR:_loop_cgal>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VCPKG_ROOT}/packages/mpfr_x64-windows/bin/mpfr-6.dll"
            "$<TARGET_FILE_DIR:_loop_cgal>"
    )
endif()

# Install for Python packages
install(TARGETS _loop_cgal
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/loop_cgal
)
