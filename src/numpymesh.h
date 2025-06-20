#ifndef NUMPYMESH_H
#define NUMPYMESH_H
#include <pybind11/numpy.h>
struct NumpyMesh {
  pybind11::array_t<double> vertices;
  pybind11::array_t<int> triangles;
};
struct NumpyPlane {
  pybind11::array_t<double> normal; // Normal vector of the plane
  pybind11::array_t<double> origin; // A point on the plane
};
#endif // NUMPYMESH_H