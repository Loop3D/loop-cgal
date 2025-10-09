# Loop-cgal

Loop-cgal is a Python package for mesh processing operations using the  CGAL (Computational Geometry Algorithms Library). It is designed for efficient geometric computations using pyvista objects.

## Features

- Python bindings for CGAL using `pybind11`.
- Current features:
    - clipping of 3D triangular surfaces
- Future features:
    - Marching cubes algorithm for isosurface extraction
    - Boolean operations on marching cube meshes.

## Installation

### Prerequisites

- C++17 or later
- Python 3.11 or later
- CGAL library
- Boost library
- CMake 3.15 or later
- pybind11
- scikit-build
- pyvista

### Build and Install

1. Clone the repository:
   ```bash
   git clone https://github.com/Loop3D/loop-cgal.git
   cd loop-cgal
   pip install .
   ```
2. Alternatively, you can install it directly from PyPI:
   ```bash
   pip install loop-cgal
   ```

### Windows using vcpkg

To install dependencies using [vcpkg](https://vcpkg.io/):

1. Install vcpkg following the [official guide](https://vcpkg.io/en/getting-started.html).
2. Install required libraries:
    ```bash
    vcpkg install yasm-tool:x86-windows cgal:x64-windows
    ```
3. Set the `VCPKG_ROOT` environment variable to point to your vcpkg installation directory.
4. Set the `CMAKE_ARGS` to include `-DCMAKE_TOOLCHAIN_FILE` and `-DVCPKG_ROOT` e.g. CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$Env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake -DVCPKG_ROOT=$Env:VCPKG_ROOT"
4. Proceed with the build and installation steps as described above.