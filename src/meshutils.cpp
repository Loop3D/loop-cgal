#include "meshutils.h"
#include "mesh.h"
#include "globals.h"
#include "sizet.h"
std::set<TriangleMesh::Edge_index>
collect_border_edges(const TriangleMesh &tm) {
  std::set<TriangleMesh::Edge_index> border_edges;
  for (const auto &halfedge : tm.halfedges()) {
    if (tm.is_border(halfedge)) {
      border_edges.insert(CGAL::edge(halfedge, tm)); // Convert halfedge to edge
    }
  }
  return border_edges;
}
double calculate_triangle_area(const std::array<double, 3> &v1,
                               const std::array<double, 3> &v2,
                               const std::array<double, 3> &v3) {
  // Validate input vertices for finite values
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(v1[i]) || !std::isfinite(v2[i]) || !std::isfinite(v3[i])) {
      return 0.0; // Return zero area for invalid vertices
    }
  }
  
  // Compute vectors
  std::array<double, 3> vec1 = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  std::array<double, 3> vec2 = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};

  // Check for zero-length vectors (degenerate triangles)
  double vec1_mag_sq = vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2];
  double vec2_mag_sq = vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2];
  
  if (vec1_mag_sq < 1e-16 || vec2_mag_sq < 1e-16) {
    return 0.0; // Degenerate triangle
  }

  // Compute cross product
  std::array<double, 3> cross_product = {vec1[1] * vec2[2] - vec1[2] * vec2[1],
                                         vec1[2] * vec2[0] - vec1[0] * vec2[2],
                                         vec1[0] * vec2[1] - vec1[1] * vec2[0]};

  // Compute magnitude of cross product with safety check
  double magnitude_squared = cross_product[0] * cross_product[0] +
                           cross_product[1] * cross_product[1] +
                           cross_product[2] * cross_product[2];
  
  if (magnitude_squared < 0.0 || !std::isfinite(magnitude_squared)) {
    return 0.0; // Invalid magnitude
  }
  
  double magnitude = std::sqrt(magnitude_squared);
  
  if (!std::isfinite(magnitude)) {
    return 0.0; // Invalid magnitude after sqrt
  }

  // Area is half the magnitude of the cross product
  return 0.5 * magnitude;
}
// ---------------------------------------------------------------------------
// Efficient export: linear‑time duplicate detection via quantised hash grid
// ---------------------------------------------------------------------------
NumpyMesh export_mesh(const TriangleMesh &tm, double area_threshold,
                      double duplicate_vertex_threshold) {
  using VIndex = TriangleMesh::Vertex_index;

  std::vector<std::array<double, 3>> vertices; // unique coords
  std::vector<std::array<int, 3>> triangles;   // face indices
  std::map<VIndex, int> vertex_index_map;      // CGAL → compact

  // —‑‑‑‑‑ 1.  Build unique‑vertex list ----------------------------------
  struct QKey {
    long long x, y, z;
    bool operator==(const QKey &o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };
  struct QHash {
    std::size_t operator()(const QKey &k) const noexcept {
      std::size_t h1 = std::hash<long long>{}(k.x);
      std::size_t h2 = std::hash<long long>{}(k.y);
      std::size_t h3 = std::hash<long long>{}(k.z);
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  const double inv = 1.0 / duplicate_vertex_threshold; // quantisation
  std::unordered_map<QKey, int, QHash> qmap;           // grid → index

  int next_idx = 0;
  for (VIndex v : tm.vertices()) {
    const auto &p = tm.point(v);
    
    // Validate vertex coordinates
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) {
      if (LoopCGAL::verbose)
        std::cout << "Warning: Non-finite vertex coordinates, skipping vertex\n";
      continue;
    }
    
    // Compute quantized key with overflow protection
    double x_scaled = p.x() * inv;
    double y_scaled = p.y() * inv;
    double z_scaled = p.z() * inv;
    
    if (!std::isfinite(x_scaled) || !std::isfinite(y_scaled) || !std::isfinite(z_scaled)) {
      if (LoopCGAL::verbose)
        std::cout << "Warning: Invalid scaled coordinates, skipping vertex\n";
      continue;
    }
    
    // Clamp to prevent overflow in llround
    const double max_coord = 1e15;
    x_scaled = std::max(-max_coord, std::min(max_coord, x_scaled));
    y_scaled = std::max(-max_coord, std::min(max_coord, y_scaled));
    z_scaled = std::max(-max_coord, std::min(max_coord, z_scaled));
    
    QKey key{llround(x_scaled), llround(y_scaled), llround(z_scaled)};

    auto it = qmap.find(key);
    if (it == qmap.end()) { // first occurrence → store
      qmap[key] = next_idx;
      vertices.push_back({p.x(), p.y(), p.z()});
      vertex_index_map[v] = next_idx++;
    } else { // duplicate → alias
      vertex_index_map[v] = it->second;
    }
  }

  if (LoopCGAL::verbose) {
    std::cout << "Vertices after remeshing: " << vertices.size() << '\n';
    std::cout << "Duplicate‑detection grid cells: " << qmap.size() << '\n';
  }

  // —‑‑‑‑‑ 2.  Build triangle list, skipping tiny faces ------------------
  for (auto f : tm.faces()) {
    std::array<int, 3> tri;
    int k = 0;
    bool valid_face = true;
    
    for (auto he : CGAL::halfedges_around_face(tm.halfedge(f), tm)) {
      if (k >= 3) {  // Safety check for face with more than 3 vertices
        if (LoopCGAL::verbose)
          std::cout << "Warning: Face has more than 3 vertices, skipping\n";
        valid_face = false;
        break;
      }
      
      VIndex target_vertex = CGAL::target(he, tm);
      auto it = vertex_index_map.find(target_vertex);
      if (it == vertex_index_map.end()) {
        if (LoopCGAL::verbose)
          std::cout << "Warning: Vertex not found in index map, skipping face\n";
        valid_face = false;
        break;
      }
      
      int vertex_idx = it->second;
      if (vertex_idx < 0 || vertex_idx >= static_cast<int>(vertices.size())) {
        if (LoopCGAL::verbose)
          std::cout << "Warning: Invalid vertex index " << vertex_idx << ", skipping face\n";
        valid_face = false;
        break;
      }
      
      tri[k++] = vertex_idx;
    }
    
    if (!valid_face || k != 3) {
      if (LoopCGAL::verbose && k != 3)
        std::cout << "Warning: Face does not have exactly 3 vertices (" << k << "), skipping\n";
      continue;
    }
    
    // Check for degenerate triangles (duplicate vertices)
    if (tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2]) {
      if (LoopCGAL::verbose)
        std::cout << "Warning: Degenerate triangle with duplicate vertices (" 
                  << tri[0] << ", " << tri[1] << ", " << tri[2] << "), skipping\n";
      continue;
    }

    double area = calculate_triangle_area(vertices[tri[0]], vertices[tri[1]],
                                          vertices[tri[2]]);

    if (area >= area_threshold)
      triangles.push_back(tri);
    else if (LoopCGAL::verbose)
      std::cout << "Skipping degenerate face (A=" << area << ")\n";
  }

  if (LoopCGAL::verbose)
    std::cout << "Kept " << triangles.size() << " triangles.\n";

  // —‑‑‑‑‑ 3.  Convert to NumPy arrays -----------------------------------
  // Safety check for empty data
  if (vertices.empty()) {
    if (LoopCGAL::verbose)
      std::cout << "Warning: No vertices to export, creating empty mesh\n";
    NumpyMesh result;
    result.vertices = pybind11::array_t<double>(std::vector<int>{0, 3});
    result.triangles = pybind11::array_t<int>(std::vector<int>{0, 3});
    return result;
  }
  
  if (triangles.empty()) {
    if (LoopCGAL::verbose)
      std::cout << "Warning: No triangles to export, creating vertex-only mesh\n";
  }
  
  // Validate data integrity before creating arrays
  for (size_t i = 0; i < vertices.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!std::isfinite(vertices[i][j])) {
        if (LoopCGAL::verbose)
          std::cout << "Warning: Non-finite vertex coordinate at vertex " << i << ", component " << j << "\n";
        // Replace with zero to prevent crash
        vertices[i][j] = 0.0;
      }
    }
  }
  
  // Validate triangle indices
  for (size_t i = 0; i < triangles.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      if (triangles[i][j] < 0 || triangles[i][j] >= static_cast<int>(vertices.size())) {
        if (LoopCGAL::verbose)
          std::cout << "Error: Invalid triangle index " << triangles[i][j] 
                    << " at triangle " << i << ", removing triangle\n";
        triangles.erase(triangles.begin() + i);
        --i; // Adjust index after removal
        break;
      }
    }
  }

  pybind11::array_t<double> vertices_array(
      {static_cast<int>(vertices.size()), 3});
  auto vbuf = vertices_array.mutable_unchecked<2>();
  for (size_t i = 0; i < vertices.size(); ++i) {
    vbuf(i, 0) = vertices[i][0];
    vbuf(i, 1) = vertices[i][1];
    vbuf(i, 2) = vertices[i][2];
  }

  pybind11::array_t<int> triangles_array(
      {static_cast<int>(triangles.size()), 3});
  auto tbuf = triangles_array.mutable_unchecked<2>();
  for (size_t i = 0; i < triangles.size(); ++i) {
    tbuf(i, 0) = triangles[i][0];
    tbuf(i, 1) = triangles[i][1];
    tbuf(i, 2) = triangles[i][2];
  }

  // —‑‑‑‑‑ 4.  Package & return ------------------------------------------
  NumpyMesh result;
  result.vertices = vertices_array;
  result.triangles = triangles_array;
  return result;
}
Exact_Mesh convert_to_exact(const TriMesh& input) {
  Exact_Mesh result;
  std::map<TriangleMesh::Vertex_index, Exact_Mesh::Vertex_index> vmap;

  for (auto v : vertices(input.get_mesh())) {
    const auto& p = input.get_mesh().point(v);
    Exact_K::Point_3 ep(p.x(), p.y(), p.z());
    vmap[v] = result.add_vertex(ep);
  }

  for (auto f : faces(input.get_mesh())) {
    std::vector<Exact_Mesh::Vertex_index> face_vertices;
    for (auto v : vertices_around_face(input.get_mesh().halfedge(f), input.get_mesh())) {
      face_vertices.push_back(vmap[v]);
    }
    result.add_face(face_vertices);
  }

  return result;
}
TriangleMesh convert_to_double_mesh(const Exact_Mesh& input) {
  TriangleMesh result;
  std::map<Exact_Mesh::Vertex_index, TriangleMesh::Vertex_index> vmap;

  for (auto v : vertices(input)) {
    const auto& p = input.point(v);
    Point dp(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
    vmap[v] = result.add_vertex(dp);
  }

  for (auto f : faces(input)) {
    std::vector<TriangleMesh::Vertex_index> face_vertices;
    for (auto v : vertices_around_face(input.halfedge(f), input)) {
      face_vertices.push_back(vmap[v]);
    }
    result.add_face(face_vertices);
  }

  return result;
}