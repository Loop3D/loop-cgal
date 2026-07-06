#ifndef MESH_H
#define MESH_H

#include <CGAL/Plane_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Vector_3.h>
#include <CGAL/property_map.h>
#include <numpymesh.h>
#include <pybind11/numpy.h>
#include <utility> // For std::pair
#include <vector>
#include "meshenums.h"
typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_K;
typedef CGAL::Surface_mesh<Exact_K::Point_3> Exact_Mesh;
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

        // Move/copy constructors — must re-bind _edge_is_constrained_map after
        // relocation, because it stores a raw pointer into _fixedEdges.
        TriMesh(TriMesh&& other) noexcept;
        TriMesh& operator=(TriMesh&& other) noexcept;
        TriMesh(const TriMesh& other);
        TriMesh& operator=(const TriMesh& other);

        // Method to cut the mesh with another surface object.
        // Returns the number of faces removed (0 = no-op / bad cut).
        int cutWithSurface(TriMesh &surface,
                            bool preserve_intersection = false,
                            bool preserve_intersection_clipper = false,
                            bool use_exact_kernel = true);

        // Clip the mesh with a halfspace defined by the plane ax+by+cz+d=0.
        // The negative side (ax+by+cz+d < 0) is kept.
        // Returns the number of faces removed (0 = no-op / bad cut).
        int clipWithPlane(double a, double b, double c, double d,
                          bool use_exact_kernel = true);

        // Corefine this mesh with another, inserting the SHARED intersection
        // polyline into BOTH meshes. After the call the two meshes carry
        // coincident vertices and edges along the intersection curve, so patches
        // taken from each (e.g. a contact cut by a fault and the matching fault
        // patch) stitch together watertight. Both meshes are mutated in place.
        // Returns the number of vertices added to this mesh by the corefinement.
        // Only the exact kernel is supported: use_exact_kernel=false throws
        // std::invalid_argument, because corefinement on the inexact predicate
        // kernel (Simple_cartesian) crashes rather than merely misbehaving.
        int corefine(TriMesh &other, bool use_exact_kernel = true);

        // Method to remesh the triangle mesh
        void remesh(bool split_long_edges,  double target_edge_length,
                    int number_of_iterations, bool protect_constraints,
                    bool relax_constraints);
        void init();
        // Cut the mesh along the isocontour property==value.
        // snap_tol: when an edge crossing lands within this fraction of an edge
        // from an existing endpoint, reuse that endpoint instead of inserting a
        // near-coincident vertex. This prevents the sliver triangles that a raw
        // insertion produces (0 = never snap, exact old behaviour).
        void cut_with_implicit_function(const std::vector<double>& property, double value, ImplicitCutMode cutmode = ImplicitCutMode::KEEP_POSITIVE_SIDE, double snap_tol = 1e-4);
        // Getters for mesh properties
        void reverseFaceOrientation();
        NumpyMesh save(double area_threshold, double duplicate_vertex_threshold);
        void add_fixed_edges(const pybind11::array_t<int> &pairs);
        double area() const;
        std::size_t n_faces() const;
        std::size_t n_vertices() const;
        pybind11::array_t<double> get_points() const;
        bool overlaps(const TriMesh& other, double bbox_tol = 1e-6) const;
        TriMesh clone() const;
        void write_to_file(const std::string& path) const;
        static TriMesh read_from_file(const std::string& path);
        const TriangleMesh& get_mesh() const { return _mesh; }
        void set_mesh(const TriangleMesh& mesh) { _mesh = mesh; }
private:
        // Internal constructor used by cutWithSurface to wrap a scaled copy.
        explicit TriMesh(TriangleMesh m) : _mesh(std::move(m)) { init(); }

        // Capture the endpoint coordinates of every edge currently in
        // _fixedEdges that is still live in _mesh. Edge_index handles are not
        // stable across clip()/collect_garbage() or the exact-kernel round
        // trip, so constraints must be tracked by geometry, not by index.
        std::vector<std::pair<Point, Point>> snapshot_constraint_coords() const;

        // Rebuild _fixedEdges (and re-bind the constrained-edge map) from a
        // coordinate snapshot: border edges of the current _mesh are always
        // re-derived, and each saved endpoint pair is re-resolved to an edge
        // descriptor in the current _mesh when both endpoints still exist and
        // are directly connected. Use after any op that invalidates indices.
        void rebuild_fixed_edges_from_coords(
            const std::vector<std::pair<Point, Point>> &saved);

        std::set<TriangleMesh::Edge_index> _fixedEdges;
        TriangleMesh _mesh; // The underlying CGAL surface mesh
        CGAL::Boolean_property_map<std::set<TriangleMesh::Edge_index>>
            _edge_is_constrained_map;
};

#endif // MESH_H