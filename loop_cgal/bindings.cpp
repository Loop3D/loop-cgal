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
     py::class_<TriMesh>(m, "TriMesh",
          "A CGAL surface mesh of triangles.\n\n"
          "NOT THREAD-SAFE. Most methods release the GIL for the duration of the "
          "CGAL call, so two Python threads touching the same TriMesh concurrently "
          "are a data race in C++ -- corrupt memory, not merely a wrong answer. That "
          "includes passing one mesh as the `other`/`surface` argument of a call "
          "running on another thread. Give each thread its own meshes, or serialise "
          "access yourself. Separate processes (the usual pattern here) are unaffected.")
         .def(py::init<const pybind11::array_t<double> &, const pybind11::array_t<int> &>(),
              py::arg("vertices"), py::arg("triangles"))
         .def(py::init([](const TriMesh& other) { return other.clone(); }),
              py::arg("other"),
              "Copy-construct a TriMesh as a deep clone of another (no array roundtrip).")
         .def("clip_with_plane", &TriMesh::clipWithPlane,
              py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
              py::arg("use_exact_kernel") = true,
              py::call_guard<py::gil_scoped_release>(),
              "Clip the mesh with the halfspace ax+by+cz+d <= 0. "
              "Uses PMP::clip(mesh, Plane_3) directly — no corefinement, no skirt construction. "
              "Returns the number of faces removed (0 = no-op).")
         .def("corefine", &TriMesh::corefine, py::arg("other"),
              py::arg("use_exact_kernel") = true,
              py::call_guard<py::gil_scoped_release>(),
              "Corefine this mesh with another, inserting the shared intersection "
              "polyline into BOTH meshes in place. NOTE this MUTATES `other` as well "
              "as self, so `other` must not be shared with a concurrent call. After the call the two meshes "
              "carry coincident vertices/edges along the intersection curve, so "
              "patches taken from each side stitch together watertight. "
              "Returns the number of vertices added to this mesh. "
              "Only the exact kernel is supported; use_exact_kernel=False raises "
              "ValueError (inexact-kernel corefinement crashes).")
         .def("cut_with_surface", &TriMesh::cutWithSurface, py::arg("surface"),
              py::arg("preserve_intersection") = false,
              py::arg("preserve_intersection_clipper") = false,
              py::arg("use_exact_kernel") = true,
              py::arg("extend") = true,
              py::call_guard<py::gil_scoped_release>(),
              "Cut mesh with a clipper surface. Returns the number of faces removed "
              "(0 indicates the clipper did not intersect or did not extend beyond the mesh). "
              "extend=True grows a skirt so a small clipper cuts all the way through; pass "
              "extend=False when the clipper already spans the domain. "
              "Mutates self and reads `surface`; neither may be touched by a concurrent call.")
         .def("remesh", &TriMesh::remesh, py::arg("split_long_edges") = true,
              py::arg("target_edge_length") = 10.0,
              py::arg("number_of_iterations") = 3,
              py::arg("protect_constraints") = true,
              py::arg("relax_constraints") = false,
              py::call_guard<py::gil_scoped_release>())
         .def("save", &TriMesh::save, py::arg("area_threshold") = 1e-6,
              py::arg("duplicate_vertex_threshold") = 1e-6)
         .def("reverse_face_orientation", &TriMesh::reverseFaceOrientation,
              "Reverse the face orientation of the mesh.")
         .def("add_fixed_edges", &TriMesh::add_fixed_edges,
              py::arg("pairs"),
              py::call_guard<py::gil_scoped_release>(),
              "Vertex index pairs defining edges to be fixed in mesh when remeshing.")
         .def("cut_with_implicit_function", &TriMesh::cut_with_implicit_function,
              py::arg("property"), py::arg("value"),py::arg("cutmode") = ImplicitCutMode::KEEP_POSITIVE_SIDE,
              py::arg("snap_tol") = 1e-4,
              py::call_guard<py::gil_scoped_release>(),
              "Cut the mesh with an implicit function defined by vertex properties. "
              "snap_tol reuses an existing vertex when a crossing lands within that "
              "fraction of an edge from it, preventing sliver triangles (0 = disable).")
         .def("_cgal_area", &TriMesh::area, "Surface area computed directly from CGAL mesh.")
         .def("_cgal_n_faces", &TriMesh::n_faces, "Number of faces in the CGAL mesh.")
         .def("_cgal_n_vertices", &TriMesh::n_vertices, "Number of vertices in the CGAL mesh.")
         .def("_cgal_points", &TriMesh::get_points, "Vertex coordinates as (n, 3) numpy array.")
         .def("overlaps", &TriMesh::overlaps, py::arg("other"), py::arg("bbox_tol") = 1e-6,
              py::call_guard<py::gil_scoped_release>(),
              "Return True if this mesh intersects another TriMesh (AABB fast-reject + triangle-level check). "
              "Reads both meshes; neither may be mutated by a concurrent call.")
         .def("does_self_intersect", &TriMesh::does_self_intersect,
              py::call_guard<py::gil_scoped_release>(),
              "Return True if the mesh intersects itself.\n\n"
              "Wraps CGAL PMP::does_self_intersect. A self-intersecting surface is "
              "outside the domain of the boolean operations - corefine, cut_with_surface "
              "and the clip paths may fail or return a corrupt result on one - so this is "
              "the check to run before handing a mesh to them, and after a remesh or a fit "
              "that could have folded it.\n\n"
              "Cost is O(n log n) via an AABB tree over the faces: cheap enough to assert "
              "on, not free in a tight loop.")
         .def("clone", &TriMesh::clone,
              "Return a deep copy of this TriMesh at the C++ level (no pyvista round-trip). "
              "Preserves the full fixed-edge set including any user-added edges.")
         .def("write_to_file", &TriMesh::write_to_file, py::arg("path"),
              "Write the mesh to a compact binary file (LCMESH format). "
              "Much faster than converting to pyvista and saving as VTK.")
         .def_static("_read_from_file", &TriMesh::read_from_file, py::arg("path"),
              "Read a mesh from a binary file written by write_to_file. "
              "Returns a _TriMesh — use TriMesh.read_from_file() from Python instead.");

} // End of PYBIND11_MODULE