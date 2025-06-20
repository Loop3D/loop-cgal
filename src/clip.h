#ifndef CLIP_H
#define CLIP_H
#include "numpymesh.h"
#include <CGAL/Plane_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Vector_3.h>
#include <pybind11/numpy.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> TriangleMesh;
typedef CGAL::Plane_3<Kernel> Plane;
typedef CGAL::Vector_3<Kernel> Vector;
std::set<TriangleMesh::Edge_index> collect_border_edges(const TriangleMesh &tm);
double calculate_triangle_area(const std::array<double, 3> &v1,
                               const std::array<double, 3> &v2,
                               const std::array<double, 3> &v3);
NumpyMesh clip_surface(NumpyMesh tm, NumpyMesh clipper,
                       double target_edge_length = 10.0,
                       bool remesh_before_clipping = true,
                       bool remesh_after_clipping = true,
                       bool remove_degenerate_faces = true,
                       double duplicate_vertex_threshold = 1e-6,
                       double area_threshold = 1e-6,
                       bool protect_constraints = true,
                       bool relax_constraints = false, bool verbose = false);
NumpyMesh clip_plane(NumpyMesh tm, NumpyPlane clipper,
                     double target_edge_length = 10.0,
                     bool remesh_before_clipping = true,
                     bool remesh_after_clipping = true,
                     bool remove_degenerate_faces = true,
                     double duplicate_vertex_threshold = 1e-6,
                     double area_threshold = 1e-6,
                     bool protect_constraints = true,
                     bool relax_constraints = false, bool verbose = false);
TriangleMesh load_mesh(NumpyMesh mesh, bool verbose = false);
Plane load_plane(NumpyPlane plane, bool verbose = false);
void refine_mesh(TriangleMesh &mesh, bool split_long_edges = true,
                 bool verbose = false, double target_edge_length = 10.0,
                 int number_of_iterations = 1, bool protect_constraints = true,
                 bool relax_constraints = false);
NumpyMesh export_mesh(const TriangleMesh &tm, double area_threshold,
                      double duplicate_vertex_threshold, bool verbose = false);
std::vector<NumpyMesh>
corefine_mesh(NumpyMesh tm1, NumpyMesh tm2, double target_edge_length = 10.0,
              double duplicate_vertex_threshold = 1e-6,
              double area_threshold = 1e-6, int number_of_iterations = 3,
              bool relax_constraints = true, bool protect_constraints = false,
              bool verbose = false);
#endif