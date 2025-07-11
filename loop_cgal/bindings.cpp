#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clip.h" // Include the API implementation
#include "mesh.h"
#include "grid.h"
#include "numpymesh.h"

namespace py = pybind11;

PYBIND11_MODULE(loop_cgal, m)
{

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
      py::class_<TriMesh>(m, "TriMesh")
          .def(py::init<const pybind11::array_t<double> &, const pybind11::array_t<int> &>(),
               py::arg("vertices"), py::arg("triangles"))
          .def("cut_with_surface", &TriMesh::cutWithSurface, py::arg("surface"),
               py::arg("verbose") = false,
               py::arg("preserve_intersection") = false,
               py::arg("preserve_intersection_clipper") = false)
          .def("remesh", &TriMesh::remesh, py::arg("split_long_edges") = true,
               py::arg("verbose") = false,
               py::arg("target_edge_length") = 10.0,
               py::arg("number_of_iterations") = 3,
               py::arg("protect_constraints") = true,
               py::arg("relax_constraints") = false)
          .def("save", &TriMesh::save, py::arg("area_threshold") = 1e-6,
               py::arg("duplicate_vertex_threshold") = 1e-6,
               py::arg("verbose") = false)
            .def("reverse_face_orientation", &TriMesh::reverseFaceOrientation,
                 "Reverse the face orientation of the mesh.");
     py::class_<Grid>(m, "Grid")
          .def(py::init<double, double, double, double, double, double, int, int, int>(),
               py::arg("origin_x"), py::arg("origin_y"), py::arg("origin_z"),
               py::arg("step_x"), py::arg("step_y"), py::arg("step_z"),
               py::arg("nsteps_x"), py::arg("nsteps_y"), py::arg("nsteps_z"))
          .def("add_scalar_field", &Grid::addScalarField, py::arg("name"),
               py::arg("values"))
          .def("extract_surface_for_cell", &Grid::extractSurfaceForCell,
               py::arg("i"), py::arg("j"), py::arg("k"), py::arg("name"),
               py::arg("isovalue"))
          .def("_extract_isosurface", &Grid::_extract_isosurface,
               py::arg("name"), py::arg("isovalue"),
               "Extract an isosurface from the grid based on a scalar field and isovalue.");
               
     


} // End of PYBIND11_MODULE