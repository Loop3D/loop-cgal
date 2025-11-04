cmake_minimum_required(VERSION 3.15)
project(loop-cgal VERSION 1.0 LANGUAGES CXX)
# --- Optional vcpkg integration ---
if(DEFINED ENV{VCPKG_ROOT})
    message(STATUS "Using VCPKG toolchain from $ENV{VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else()
    message(STATUS "VCPKG_ROOT not defined — using system packages.")
endif()
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

if(WIN32)
    set_target_properties(_loop_cgal PROPERTIES PREFIX "" SUFFIX ".pyd")
else()
    set_target_properties(_loop_cgal PROPERTIES PREFIX "" SUFFIX ".so")
endif()

# Windows: copy required DLLs from VCPKG_ROOT


if(WIN32)
    install(CODE "
        file(INSTALL
            DESTINATION \"\${CMAKE_INSTALL_PREFIX}/loop_cgal\"
            TYPE FILE
            FILES
                \"${VCPKG_ROOT}/packages/gmp_x64-windows/bin/gmp-10.dll\"
                \"${VCPKG_ROOT}/packages/mpfr_x64-windows/bin/mpfr-6.dll\"
        )
        message(\"[CMake Install] Copied gmp-10.dll and mpfr-6.dll to install directory.\")
    ")
endif()



# Install for Python packages
install(TARGETS _loop_cgal
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/loop_cgal
)
