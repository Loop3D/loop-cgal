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

  for (ssize_t i = 0; i < tris.shape(0); ++i)
  {
    int v0 = tris(i, 0);
    int v1 = tris(i, 1);
    int v2 = tris(i, 2);

    // Check that all vertex indices are valid
    if (v0 < 0 || v0 >= vertex_indices.size() || v1 < 0 ||
        v1 >= vertex_indices.size() || v2 < 0 || v2 >= vertex_indices.size())
    {
      std::cerr << "Warning: Triangle " << i << " has invalid vertex indices: ("
                << v0 << ", " << v1 << ", " << v2 << "). Skipping."
                << std::endl;
      continue;
    }

    // Check for degenerate triangles
    if (v0 == v1 || v1 == v2 || v0 == v2)
    {
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
    if (!_mesh.is_valid(edge)) // Check if the halfedge is valid
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
  PMP::remove_isolated_vertices(_mesh);
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose))
  {
    std::cerr << "Mesh is not valid before reversing face orientations."
              << std::endl;
    return;
  }
  PMP::reverse_face_orientations(_mesh);
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose))
  {
    std::cerr << "Mesh is not valid after reversing face orientations."
              << std::endl;
  }
}

void TriMesh::cutWithSurface(TriMesh &clipper,
                             bool preserve_intersection,
                             bool preserve_intersection_clipper,
                            bool use_exact_kernel)
{

  if (LoopCGAL::verbose)
  {
    std::cout << "Cutting mesh with surface." << std::endl;
  }

  // Validate input meshes
  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose))
  {
    std::cerr << "Error: Source mesh is invalid!" << std::endl;
    return;
  }

  if (!CGAL::is_valid_polygon_mesh(clipper._mesh, LoopCGAL::verbose))
  {
    std::cerr << "Error: Clipper mesh is invalid!" << std::endl;
    return;
  }

  if (_mesh.number_of_vertices() == 0 || _mesh.number_of_faces() == 0)
  {
    std::cerr << "Error: Source mesh is empty!" << std::endl;
    return;
  }

  if (clipper._mesh.number_of_vertices() == 0 ||
      clipper._mesh.number_of_faces() == 0)
  {
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

    try
    {
      // bool flag =
      //     PMP::clip(_mesh, clipper._mesh, CGAL::parameters::clip_volume(false));
      bool flag = false;
      try
      {
        if (use_exact_kernel){
          Exact_Mesh exact_clipper = convert_to_exact(clipper);
          Exact_Mesh exact_mesh = convert_to_exact(*this);
          flag = PMP::clip(exact_mesh, exact_clipper, CGAL::parameters::clip_volume(false));
        set_mesh(convert_to_double_mesh(exact_mesh));
        }
        else{
          flag = PMP::clip(_mesh, clipper._mesh, CGAL::parameters::clip_volume(false));
        }
        
      }
      catch (const std::exception &e)
      {
        std::cerr << "Corefinement failed: " << e.what() << std::endl;
      }
      if (!flag)
      {
        std::cerr << "Warning: Clipping operation failed." << std::endl;
      }
      else
      {
        if (LoopCGAL::verbose)
        {
          std::cout << "Clipping successful. Result has "
                    << _mesh.number_of_vertices() << " vertices and "
                    << _mesh.number_of_faces() << " faces." << std::endl;
        }
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during clipping: " << e.what() << std::endl;
    }
  }
  else
  {
    if (LoopCGAL::verbose)
    {
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

void TriMesh::cut_with_implicit_function(const std::vector<double> &property, double value, ImplicitCutMode cutmode)
{
  std::cout << "Cutting mesh with implicit function at value " << value << std::endl;
  std::cout << "Mesh has " << _mesh.number_of_vertices() << " vertices and "
            << _mesh.number_of_faces() << " faces." << std::endl;
  std::cout << "Property size: " << property.size() << std::endl;
  if (property.size() != _mesh.number_of_vertices())
  {
    std::cerr << "Error: Property size does not match number of vertices." << std::endl;
    return;
  }
  // Create a property map for vertex properties
  typedef boost::property_map<TriangleMesh, boost::vertex_index_t>::type VertexIndexMap;
  VertexIndexMap vim = get(boost::vertex_index, _mesh);
  std::vector<double> vertex_properties(_mesh.number_of_vertices());
  for (auto v : _mesh.vertices())
  {
    vertex_properties[vim[v]] = property[vim[v]];
  }
  auto property_map = boost::make_iterator_property_map(
      vertex_properties.begin(), vim);

  // Build arrays similar to Python helper build_surface_arrays
  // We'll need: edges map (ordered pair -> edge index), tri2edge mapping, edge->endpoints array

  // Map ordered vertex pair to an integer edge id
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> edge_index_map;
  std::vector<std::pair<std::size_t, std::size_t>> edge_array; // endpoints
  std::vector<std::array<std::size_t, 3>> tri_array;           // triangles by vertex indices

  // Fill tri_array
  for (auto f : _mesh.faces())
  {
    std::array<std::size_t, 3> tri;
    int i = 0;
    for (auto v : vertices_around_face(_mesh.halfedge(f), _mesh))
    {
      tri[i++] = vim[v];
    }
    tri_array.push_back(tri);
  }

  // Helper to get or create edge index
  auto get_edge_id = [&](std::size_t a, std::size_t b)
  {
    if (a > b)
      std::swap(a, b);
    auto key = std::make_pair(a, b);
    auto it = edge_index_map.find(key);
    if (it != edge_index_map.end())
      return it->second;
    std::size_t id = edge_array.size();
    edge_array.push_back(key);
    edge_index_map[key] = id;
    return id;
  };

  std::vector<std::array<std::size_t, 3>> tri2edge(tri_array.size());
  for (std::size_t ti = 0; ti < tri_array.size(); ++ti)
  {
    auto &tri = tri_array[ti];
    tri2edge[ti][0] = get_edge_id(tri[1], tri[2]);
    tri2edge[ti][1] = get_edge_id(tri[2], tri[0]);
    tri2edge[ti][2] = get_edge_id(tri[0], tri[1]);
  }

  // Determine active triangles (all > value OR all < value OR any nan)
  std::vector<char> active(tri_array.size(), 0);
  for (std::size_t ti = 0; ti < tri_array.size(); ++ti)
  {
    auto &tri = tri_array[ti];
    double v1 = vertex_properties[tri[0]];
    double v2 = vertex_properties[tri[1]];
    double v3 = vertex_properties[tri[2]];
    bool nan1 = std::isnan(v1);
    bool nan2 = std::isnan(v2);
    bool nan3 = std::isnan(v3);
    if (nan1 || nan2 || nan3)
    {
      active[ti] = 1;
      continue;
    }
    if ((v1 > value && v2 > value && v3 > value) || (v1 < value && v2 < value && v3 < value))
    {
      active[ti] = 1;
    }
  }

  // Prepare new vertex list and new triangles similar to python
  std::vector<Point> verts;
  verts.reserve(_mesh.number_of_vertices());
  for (auto v : _mesh.vertices())
    verts.push_back(_mesh.point(v));
  std::vector<Point> newverts = verts;
  std::vector<double> newvals = vertex_properties;

  std::map<std::size_t, std::size_t> new_point_on_edge;
  std::vector<std::array<std::size_t, 3>> newtris(tri_array.begin(), tri_array.end());
  if (LoopCGAL::verbose)
  {
    std::cout << "Starting main loop over " << tri_array.size() << " triangles." << std::endl;
  }
  for (std::size_t t = 0; t < tri_array.size(); ++t)
  {
    if (active[t])
    {
      continue;
    }
    auto tri = tri_array[t];
    // if all > value skip (hanging_wall in python)
    if (vertex_properties[tri[0]] > value && vertex_properties[tri[1]] > value && vertex_properties[tri[2]] > value)
      continue;
    // for each edge of tri, check if edge crosses
    for (auto eid : tri2edge[t])
    {
      auto ends = edge_array[eid];
      double f0 = vertex_properties[ends.first];
      double f1 = vertex_properties[ends.second];
      if ((f0 > value && f1 > value) || (f0 < value && f1 < value))
      {
        if (LoopCGAL::verbose)
        {
          std::cout << "Edge " << ends.first << "-" << ends.second << " does not cross the value " << value << std::endl;
        }
        continue;
      }
      double denom = (f1 - f0);
      double ratio = 0.0;
      if (std::abs(denom) < 1e-12)
        ratio = 0.5;
      else
        ratio = (value - f0) / denom;
      Point p0 = verts[ends.first];
      Point p1 = verts[ends.second];
      Point np = Point(p0.x() + ratio * (p1.x() - p0.x()), p0.y() + ratio * (p1.y() - p0.y()), p0.z() + ratio * (p1.z() - p0.z()));
      newverts.push_back(np);
      newvals.push_back(value);
      new_point_on_edge[eid] = newverts.size() - 1;
    }

    double v1 = vertex_properties[tri[0]];
    double v2 = vertex_properties[tri[1]];
    double v3 = vertex_properties[tri[2]];
    // replicate python cases
    // convert tri to vector of 3 original indices and 2 new points
    std::array<std::size_t, 5> extended = {tri[0], tri[1], tri[2], 0, 0};
    // retrieve relevant edges indices
    std::size_t e01 = edge_index_map[std::make_pair(std::min(tri[0], tri[1]), std::max(tri[0], tri[1]))];
    std::size_t e12 = edge_index_map[std::make_pair(std::min(tri[1], tri[2]), std::max(tri[1], tri[2]))];
    std::size_t e20 = edge_index_map[std::make_pair(std::min(tri[2], tri[0]), std::max(tri[2], tri[0]))];
    // Get new points where available
    std::size_t np_e01 = new_point_on_edge.count(e01) ? new_point_on_edge[e01] : SIZE_MAX;
    std::size_t np_e12 = new_point_on_edge.count(e12) ? new_point_on_edge[e12] : SIZE_MAX;
    std::size_t np_e20 = new_point_on_edge.count(e20) ? new_point_on_edge[e20] : SIZE_MAX;
    // Helper to append triangle
    auto append_tri = [&](std::array<std::size_t, 3> tarr)
    { newtris.push_back(tarr); };

    // CASE 1: v1 > value and v2 > value and v3<value
    if (v1 > value && v2 > value && v3 < value)
    {
      std::size_t p1 = np_e12;
      std::size_t p2 = np_e20;
      extended[3] = p1;
      extended[4] = p2;
      std::array<std::size_t, 3> m1 = {extended[0], extended[1], extended[3]};
      std::array<std::size_t, 3> m2 = {extended[0], extended[3], extended[4]};
      std::array<std::size_t, 3> m3 = {extended[4], extended[3], extended[2]};
      newtris[t] = m1;
      append_tri(m2);
      append_tri(m3);
      if (LoopCGAL::verbose)
      {
        std::cout << "CASE 1 executed" << std::endl;
      }
      continue;
    }
    // CASE 2
    if (v1 > value && v2 < value && v3 > value)
    {
      std::size_t p1 = np_e01;
      std::size_t p2 = np_e12;
      extended[3] = p1;
      extended[4] = p2;
      std::array<std::size_t, 3> m1 = {extended[0], extended[3], extended[2]};
      std::array<std::size_t, 3> m2 = {extended[3], extended[4], extended[2]};
      std::array<std::size_t, 3> m3 = {extended[3], extended[1], extended[4]};
      newtris[t] = m1;
      append_tri(m2);
      append_tri(m3);
      if (LoopCGAL::verbose)
      {
        std::cout << "CASE 2 executed" << std::endl;
      }
      continue;
    }
    // CASE 3
    if (v1 < value && v2 > value && v3 > value)
    {
      std::size_t p1 = np_e01;
      std::size_t p2 = np_e20;
      extended[3] = p1;
      extended[4] = p2;
      std::array<std::size_t, 3> m1 = {extended[0], extended[3], extended[4]};
      std::array<std::size_t, 3> m2 = {extended[3], extended[1], extended[2]};
      std::array<std::size_t, 3> m3 = {extended[4], extended[3], extended[2]};
      newtris[t] = m1;
      append_tri(m2);
      append_tri(m3);
      if (LoopCGAL::verbose)
      {
        std::cout << "CASE 3 executed" << std::endl;
      }
      continue;
    }
    // CASE 5
    if (v1 < value && v2 < value && v3 > value)
    {
      std::size_t p1 = np_e12;
      std::size_t p2 = np_e20;
      extended[3] = p1;
      extended[4] = p2;
      std::array<std::size_t, 3> m1 = {extended[0], extended[1], extended[3]};
      std::array<std::size_t, 3> m2 = {extended[0], extended[3], extended[4]};
      std::array<std::size_t, 3> m3 = {extended[4], extended[3], extended[2]};
      newtris[t] = m1;
      append_tri(m2);
      append_tri(m3);
      if (LoopCGAL::verbose)
      {
        std::cout << "CASE 5 executed" << std::endl;
      }
      continue;
    }
    // CASE 6
    if (v1 < value && v2 > value && v3 < value)
    {
      std::size_t p1 = np_e01;
      std::size_t p2 = np_e12;
      extended[3] = p1;
      extended[4] = p2;
      std::array<std::size_t, 3> m1 = {extended[0], extended[3], extended[2]};
      std::array<std::size_t, 3> m2 = {extended[3], extended[4], extended[2]};
      std::array<std::size_t, 3> m3 = {extended[3], extended[1], extended[4]};
      newtris[t] = m1;
      append_tri(m2);
      append_tri(m3);
      if (LoopCGAL::verbose)
      {
        std::cout << "CASE 6 executed" << std::endl;
      }
      continue;
    }
    // CASE 7
    if (v1 > value && v2 < value && v3 < value)
    {
      std::size_t p1 = np_e01;
      std::size_t p2 = np_e20;
      extended[3] = p1;
      extended[4] = p2;
      std::array<std::size_t, 3> m1 = {extended[0], extended[3], extended[4]};
      std::array<std::size_t, 3> m2 = {extended[3], extended[2], extended[4]};
      std::array<std::size_t, 3> m3 = {extended[3], extended[1], extended[2]};
      newtris[t] = m1;
      append_tri(m2);
      append_tri(m3);
      if (LoopCGAL::verbose)
      {
        std::cout << "CASE 7 executed" << std::endl;
      }
      continue;
    }
  }

  // Build new CGAL mesh from newverts and newtris
  TriangleMesh newmesh;
  std::vector<TriangleMesh::Vertex_index> new_vhandles;
  new_vhandles.reserve(newverts.size());
  for (auto &p : newverts)
    new_vhandles.push_back(newmesh.add_vertex(p));
  for (auto &tri : newtris)
  {
    // skip degenerate
    if (tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2])
      continue;
    if (ImplicitCutMode::KEEP_NEGATIVE_SIDE == cutmode)
    {
      double v0 = newvals[tri[0]];
      double v1 = newvals[tri[1]];
      double v2 = newvals[tri[2]];
      if (v0 > value && v1 > value && v2 > value)
      {
        continue;
      }
    }
    if (ImplicitCutMode::KEEP_POSITIVE_SIDE == cutmode)
    {
      double v0 = newvals[tri[0]];
      double v1 = newvals[tri[1]];
      double v2 = newvals[tri[2]];
      if (v0 < value && v1 < value && v2 < value)
      {
        continue;
      }
    }
    newmesh.add_face(new_vhandles[tri[0]], new_vhandles[tri[1]], new_vhandles[tri[2]]);
  }

  // Replace internal mesh
  _mesh = std::move(newmesh);
}
