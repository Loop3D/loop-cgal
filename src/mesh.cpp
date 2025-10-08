#include "mesh.h"
#include "meshutils.h"
#include "globals.h"
#include "sizet.h"
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/merge_border_vertices.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/properties.h>
#include <CGAL/version.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace PMP = CGAL::Polygon_mesh_processing;

TriMesh::TriMesh(const std::vector<std::vector<int>> &triangles,
                 const std::vector<std::pair<double, double>> &vertices)
{

  std::vector<TriangleMesh::Vertex_index> vertex_indices;
  if (LoopCGAL::verbose)
  {
    std::cout << "Loading mesh with " << vertices.size() << " vertices and "
              << triangles.size() << " triangles." << std::endl;
  }

  // Assemble CGAL mesh objects from numpy/pybind11 arrays
  for (ssize_t i = 0; i < vertices.size(); ++i)
  {
    vertex_indices.push_back(
        _mesh.add_vertex(Point(vertices[i].first, vertices[i].second, 0.0)));
  }
  for (ssize_t i = 0; i < triangles.size(); ++i)
  {
    _mesh.add_face(vertex_indices[triangles[i][0]],
                   vertex_indices[triangles[i][1]],
                   vertex_indices[triangles[i][2]]);
  }

  if (LoopCGAL::verbose)
  {
    std::cout << "Loaded mesh with " << _mesh.number_of_vertices()
              << " vertices and " << _mesh.number_of_faces() << " faces."
              << std::endl;
  }
  init();
}

TriMesh::TriMesh(const pybind11::array_t<double> &vertices,
                 const pybind11::array_t<int> &triangles)
{
  auto verts = vertices.unchecked<2>();
  auto tris = triangles.unchecked<2>();
  std::vector<TriangleMesh::Vertex_index> vertex_indices;

  for (ssize_t i = 0; i < verts.shape(0); ++i)
  {
    vertex_indices.push_back(
        _mesh.add_vertex(Point(verts(i, 0), verts(i, 1), verts(i, 2))));
  }

  for (ssize_t i = 0; i < tris.shape(0); ++i) {
    int v0 = tris(i, 0);
    int v1 = tris(i, 1);
    int v2 = tris(i, 2);

    // Check that all vertex indices are valid
    if (v0 < 0 || v0 >= vertex_indices.size() || v1 < 0 ||
        v1 >= vertex_indices.size() || v2 < 0 || v2 >= vertex_indices.size()) {
      std::cerr << "Warning: Triangle " << i << " has invalid vertex indices: ("
                << v0 << ", " << v1 << ", " << v2 << "). Skipping."
                << std::endl;
      continue;
    }

    // Check for degenerate triangles
    if (v0 == v1 || v1 == v2 || v0 == v2) {
      std::cerr << "Warning: Triangle " << i << " is degenerate: (" << v0
                << ", " << v1 << ", " << v2 << "). Skipping." << std::endl;
      continue;
    }

    _mesh.add_face(vertex_indices[v0], vertex_indices[v1], vertex_indices[v2]);
  }
  for (ssize_t i = 0; i < tris.shape(0); ++i)
  {
    _mesh.add_face(vertex_indices[tris(i, 0)], vertex_indices[tris(i, 1)],
                   vertex_indices[tris(i, 2)]);
  }
  if (LoopCGAL::verbose)
  {
    std::cout << "Loaded mesh with " << _mesh.number_of_vertices()
              << " vertices and " << _mesh.number_of_faces() << " faces."
              << std::endl;
  }

  init();
}

void TriMesh::init()
{
  _fixedEdges = collect_border_edges(_mesh);


  if (LoopCGAL::verbose)
  {
    std::cout << "Found " << _fixedEdges.size() << " fixed edges." << std::endl;
  }
  _edge_is_constrained_map = CGAL::make_boolean_property_map(_fixedEdges);
}

void TriMesh::add_fixed_edges(const pybind11::array_t<int> &pairs)
{
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose))
  {
    std::cerr << "Mesh is not valid!" << std::endl;
  }
  // Convert std::set<std::array<int, 2>> to std::set<TriangleMesh::Edge_index>
  auto pairs_buf = pairs.unchecked<2>();

  for (ssize_t i = 0; i < pairs_buf.shape(0); ++i)
  {
    TriangleMesh::Vertex_index v0 = TriangleMesh::Vertex_index(pairs_buf(i, 1));
    TriangleMesh::Vertex_index v1 = TriangleMesh::Vertex_index(pairs_buf(i, 0));
    if (!_mesh.is_valid(v0) || !_mesh.is_valid(v1))
    {
      std::cerr << "Invalid vertex indices: (" << v0 << ", " << v1 << ")"
                << std::endl;
      continue; // Skip invalid vertex pairs
    }
    TriangleMesh::Halfedge_index edge =
        _mesh.halfedge(TriangleMesh::Vertex_index(pairs_buf(i, 0)),
                       TriangleMesh::Vertex_index(pairs_buf(i, 1)));
    if (edge == TriangleMesh::null_halfedge())
    {
      std::cerr << "Half-edge is null for vertices (" << v1 << ", " << v0 << ")"
                << std::endl;
      continue;
    }
    if (!_mesh.is_valid(edge))  // Check if the halfedge is valid
    {
      std::cerr << "Invalid half-edge for vertices (" << v0 << ", " << v1 << ")"
                << std::endl;
      continue; // Skip invalid edges
    }
    TriangleMesh::Edge_index e = _mesh.edge(edge);

    _fixedEdges.insert(e);
    //     if (e.is_valid()) {
    //         _fixedEdges.insert(e);
    //     } else {
    //         std::cerr << "Warning: Edge (" << edge[0] << ", " << edge[1] <<
    //         ") is not valid in the mesh." << std::endl;
    //     }
  }
  // // Update the property map with the new fixed edges
  _edge_is_constrained_map = CGAL::make_boolean_property_map(_fixedEdges);
}
void TriMesh::remesh(bool split_long_edges,
                     double target_edge_length, int number_of_iterations,
                     bool protect_constraints, bool relax_constraints)

{

  // ------------------------------------------------------------------
  // 0.  Guard‑rail: sensible target length w.r.t. bbox
  // ------------------------------------------------------------------
  CGAL::Bbox_3 bb = PMP::bbox(_mesh);
  const double bbox_diag = std::sqrt(CGAL::square(bb.xmax() - bb.xmin()) +
                                     CGAL::square(bb.ymax() - bb.ymin()) +
                                     CGAL::square(bb.zmax() - bb.zmin()));
  PMP::remove_isolated_vertices(_mesh);
  if (target_edge_length < 1e-4 * bbox_diag)
  {
    if (LoopCGAL::verbose)
      std::cout << "  ! target_edge_length (" << target_edge_length
                << ") too small – skipping remesh\n";
    return;
  }

  // ------------------------------------------------------------------
  // 1.  Quick diagnostics
  // ------------------------------------------------------------------
  double min_e = std::numeric_limits<double>::max(), max_e = 0.0;
  for (auto e : _mesh.edges())
  {
    const double l = PMP::edge_length(e, _mesh);
    min_e = std::min(min_e, l);
    max_e = std::max(max_e, l);
  }
  if (LoopCGAL::verbose)
  {
    std::cout << "      edge length range: [" << min_e << ", " << max_e
              << "]  target = " << target_edge_length << '\n';
  }
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose) && LoopCGAL::verbose)
  {
    std::cout << "      ! mesh is not a valid polygon mesh\n";
  }
  // ------------------------------------------------------------------
  // 2.  Abort when self‑intersections remain
  // ------------------------------------------------------------------
  // std::vector<std::pair<face_descriptor, face_descriptor>> overlaps;
  // PMP::self_intersections(mesh, std::back_inserter(overlaps));
  // if (!overlaps.empty()) {
  //   if (verbose)
  //     std::cout << "      --> " << overlaps.size()
  //               << " self‑intersections – remesh skipped\n";
  //   return;
  // }

  // ------------------------------------------------------------------
  // 3.  “Tiny patch” bailout: only split long edges
  // ------------------------------------------------------------------
  // const std::size_t n_faces = _mesh.number_of_faces();
  // if (n_faces < 40)
  // {
  //     if (split_long_edges)
  //         PMP::split_long_edges(edges(_mesh), target_edge_length, _mesh);
  //     if (verbose)
  //         std::cout << "      → tiny patch (" << n_faces
  //                   << " faces) – isotropic remesh skipped\n";
  //     return;
  // }

  // ------------------------------------------------------------------
  // 4.  Normal isotropic remeshing loop
  // ------------------------------------------------------------------
  // Convert _fixedEdges to a compatible property map

  // Update remeshing calls to use the property map
  if (split_long_edges)
  {
    if (LoopCGAL::verbose)
      std::cout << "Splitting long edges before remeshing.\n";
    PMP::split_long_edges(
        edges(_mesh), target_edge_length, _mesh,
        CGAL::parameters::edge_is_constrained_map(_edge_is_constrained_map));
  }
  for (int iter = 0; iter < number_of_iterations; ++iter)
  {
    if (split_long_edges)
      if (LoopCGAL::verbose)
        std::cout << "Splitting long edges in iteration " << iter + 1 << ".\n";
    PMP::split_long_edges(
        edges(_mesh), target_edge_length, _mesh,
        CGAL::parameters::edge_is_constrained_map(_edge_is_constrained_map));
    if (LoopCGAL::verbose)
      std::cout << "Remeshing iteration " << iter + 1 << " of "
                << number_of_iterations << ".\n";
    PMP::isotropic_remeshing(
        faces(_mesh), target_edge_length, _mesh,
        CGAL::parameters::number_of_iterations(1) // one sub‑iteration per loop
            .edge_is_constrained_map(_edge_is_constrained_map)
            .protect_constraints(protect_constraints)
            .relax_constraints(relax_constraints));
  }

  if (LoopCGAL::verbose)
  {
    std::cout << "Refined mesh → " << _mesh.number_of_vertices() << " V, "
              << _mesh.number_of_faces() << " F\n";
  }
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose) && LoopCGAL::verbose)
    std::cout << "      ! mesh is not a valid polygon mesh after remeshing\n";
}

void TriMesh::reverseFaceOrientation()
{
  // Reverse the face orientation of the mesh
  PMP::reverse_face_orientations(_mesh);
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose))
  {
    std::cerr << "Mesh is not valid after reversing face orientations."
              << std::endl;
  }
  
}

void TriMesh::cutWithSurface(TriMesh &clipper,
                             bool preserve_intersection,
                             bool preserve_intersection_clipper)
{
  if (LoopCGAL::verbose)
  {
    std::cout << "Cutting mesh with surface." << std::endl;
  }

  // Validate input meshes
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose)) {
    std::cerr << "Error: Source mesh is invalid!" << std::endl;
    return;
  }

  if (!CGAL::is_valid_polygon_mesh(clipper._mesh, LoopCGAL::verbose)) {
    std::cerr << "Error: Clipper mesh is invalid!" << std::endl;
    return;
  }

  if (_mesh.number_of_vertices() == 0 || _mesh.number_of_faces() == 0) {
    std::cerr << "Error: Source mesh is empty!" << std::endl;
    return;
  }

  if (clipper._mesh.number_of_vertices() == 0 ||
      clipper._mesh.number_of_faces() == 0) {
    std::cerr << "Error: Clipper mesh is empty!" << std::endl;
    return;
  }

  bool intersection = PMP::do_intersect(_mesh, clipper._mesh);
  if (intersection)
  {
    // Clip tm with clipper
    if (LoopCGAL::verbose)
    {
      std::cout << "Clipping tm with clipper." << std::endl;
    }

    try {
      bool flag =
          PMP::clip(_mesh, clipper._mesh, CGAL::parameters::clip_volume(false));

      if (!flag) {
        std::cerr << "Warning: Clipping operation failed." << std::endl;
      } else {
        if (LoopCGAL::verbose) {
          std::cout << "Clipping successful. Result has "
                    << _mesh.number_of_vertices() << " vertices and "
                    << _mesh.number_of_faces() << " faces." << std::endl;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error during clipping: " << e.what() << std::endl;
    }
  } else {
    if (LoopCGAL::verbose) {
      std::cout << "Meshes do not intersect. No clipping performed."
                << std::endl;
    }
  }
}

NumpyMesh TriMesh::save(double area_threshold,
                        double duplicate_vertex_threshold)
{
  return export_mesh(_mesh, area_threshold, duplicate_vertex_threshold);
}