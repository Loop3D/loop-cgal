#ifndef MESH_H
#define MESH_H

#include <CGAL/Plane_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Vector_3.h>
#include <CGAL/property_map.h>
#include <numpymesh.h>
#include <pybind11/numpy.h>
#include <utility> // For std::pair
#include <vector>
#include "meshenums.h"
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> TriangleMesh;
typedef CGAL::Plane_3<Kernel> Plane;
typedef CGAL::Vector_3<Kernel> Vector;
class TriMesh
{
public:
        // Constructor
        TriMesh(const std::vector<std::vector<int>> &triangles,
                const std::vector<std::pair<double, double>> &vertices);
        TriMesh(const pybind11::array_t<double> &vertices,
                const pybind11::array_t<int> &triangles);

        // Method to cut the mesh with another surface object
        void cutWithSurface(TriMesh &surface, 
                            bool preserve_intersection = false,
                            bool preserve_intersection_clipper = false);

        // Method to remesh the triangle mesh
        void remesh(bool split_long_edges,  double target_edge_length,
                    int number_of_iterations, bool protect_constraints,
                    bool relax_constraints);
        void init();
        void cut_with_implicit_function(const std::vector<double>& property, double value, ImplicitCutMode cutmode = ImplicitCutMode::KEEP_POSITIVE_SIDE);
        // Getters for mesh properties
        void reverseFaceOrientation();
        NumpyMesh save(double area_threshold, double duplicate_vertex_threshold);
        void add_fixed_edges(const pybind11::array_t<int> &pairs);

private:
        std::set<TriangleMesh::Edge_index> _fixedEdges;
        TriangleMesh _mesh; // The underlying CGAL surface mesh
        CGAL::Boolean_property_map<std::set<TriangleMesh::Edge_index>>
            _edge_is_constrained_map;
};

#endif // MESH_HANDLER_H