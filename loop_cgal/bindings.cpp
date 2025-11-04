#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mesh.h" // include trimesh class with clipping and remeshing methods
#include "numpymesh.h"
#include "globals.h" // Include the global verbose flag
namespace py = pybind11;

PYBIND11_MODULE(_loop_cgal, m)
{
     m.attr("verbose") = &LoopCGAL::verbose; // Expose the global verbose flag
     m.def("set_verbose", &LoopCGAL::set_verbose, "Set the verbose flag");
     py::class_<NumpyMesh>(m, "NumpyMesh")
         .def(py::init<>())
         .def_readwrite("vertices", &NumpyMesh::vertices)
         .def_readwrite("triangles", &NumpyMesh::triangles);
     py::enum_<ImplicitCutMode>(m, "ImplicitCutMode")
         .value("PRESERVE_INTERSECTION", ImplicitCutMode::PRESERVE_INTERSECTION)
         .value("KEEP_POSITIVE_SIDE", ImplicitCutMode::KEEP_POSITIVE_SIDE)
         .value("KEEP_NEGATIVE_SIDE", ImplicitCutMode::KEEP_NEGATIVE_SIDE)
         .export_values();
     py::class_<TriMesh>(m, "TriMesh")
         .def(py::init<const pybind11::array_t<double> &, const pybind11::array_t<int> &>(),
              py::arg("vertices"), py::arg("triangles"))
         .def("cut_with_surface", &TriMesh::cutWithSurface, py::arg("surface"),
              py::arg("preserve_intersection") = false,
              py::arg("preserve_intersection_clipper") = false,
              py::arg("use_exact_kernel") = true)
         .def("remesh", &TriMesh::remesh, py::arg("split_long_edges") = true,
              py::arg("target_edge_length") = 10.0,
              py::arg("number_of_iterations") = 3,
              py::arg("protect_constraints") = true,
              py::arg("relax_constraints") = false)
         .def("save", &TriMesh::save, py::arg("area_threshold") = 1e-6,
              py::arg("duplicate_vertex_threshold") = 1e-6)
         .def("reverse_face_orientation", &TriMesh::reverseFaceOrientation,
              "Reverse the face orientation of the mesh.")
         .def("add_fixed_edges", &TriMesh::add_fixed_edges,
              py::arg("pairs"),
              "Vertex index pairs defining edges to be fixed in mesh when remeshing.")
         .def("cut_with_implicit_function", &TriMesh::cut_with_implicit_function,
              py::arg("property"), py::arg("value"),py::arg("cutmode") = ImplicitCutMode::KEEP_POSITIVE_SIDE,
              "Cut the mesh with an implicit function defined by vertex properties.");

} // End of PYBIND11_MODULE