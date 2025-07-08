# Minimum CMake version
cmake_minimum_required(VERSION 3.15)

# Project name and version
project(loop-cgal VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CGAL package
find_package(CGAL REQUIRED)

# Add the include directory for your project
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add the executable for standalone testing
# add_executable(loop-cgal-exe
#     src/marching_cubes.cpp
# )

# Link CGAL to the executable
# target_link_libraries(loop-cgal-exe PRIVATE CGAL::CGAL)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add the Python module
add_library(loop_cgal MODULE
    loop_cgal/bindings.cpp
    src/clip.cpp
    src/mesh.cpp
    src/meshutils.cpp
    src/grid.cpp
    src/marchingcube.cpp
    src/topologymanager.cpp
)
target_link_libraries(loop_cgal PRIVATE pybind11::module CGAL::CGAL)
target_include_directories(loop_cgal PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(loop_cgal PROPERTIES PREFIX "" SUFFIX ".so")
# Install the Python module to the correct location
install(TARGETS loop_cgal
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/loop_cgal
)