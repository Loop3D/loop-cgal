#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clip.h" // Include the API implementation

namespace py = pybind11;

PYBIND11_MODULE(loop_cgal, m) {

  m.def("clip_surface", &clip_surface, py::arg("tm"), py::arg("clipper"),
        py::arg("target_edge_length") = 10.0,
        py::arg("remesh_before_clipping") = true,
        py::arg("remesh_after_clipping") = true,
        py::arg("remove_degenerate_faces") = true,
        py::arg("duplicate_vertex_threshold") = 1e-6,
        py::arg("area_threshold") = 1e-6,
        py::arg("protect_constraints") = false,
        py::arg("relax_constraints") = true, py::arg("verbose") = false,
        "Clip one surface with another.");
  m.def("clip_plane", &clip_plane, py::arg("tm"), py::arg("clipper"),
        py::arg("target_edge_length") = 10.0,
        py::arg("remesh_before_clipping") = true,
        py::arg("remesh_after_clipping") = true,
        py::arg("remove_degenerate_faces") = true,
        py::arg("duplicate_vertex_threshold") = 1e-6,
        py::arg("area_threshold") = 1e-6,
        py::arg("protect_constraints") = false,
        py::arg("relax_constraints") = true, py::arg("verbose") = false,

        "Clip a surface with a plane.");
  m.def("corefine_mesh", &corefine_mesh, py::arg("tm1"), py::arg("tm2"),
        py::arg("target_edge_length") = 10.0,
        py::arg("duplicate_vertex_threshold") = 1e-6,
        py::arg("area_threshold") = 1e-6, py::arg("number_of_iterations") = 3,
        py::arg("relax_constraints") = true,
        py::arg("protect_constraints") = false, py::arg("verbose") = false,
        "Corefine two meshes.");
  py::class_<NumpyMesh>(m, "NumpyMesh")
      .def(py::init<>())
      .def_readwrite("vertices", &NumpyMesh::vertices)
      .def_readwrite("triangles", &NumpyMesh::triangles);
  py::class_<NumpyPlane>(m, "NumpyPlane")
      .def(py::init<>())
      .def_readwrite("normal", &NumpyPlane::normal)
      .def_readwrite("origin", &NumpyPlane::origin);
}