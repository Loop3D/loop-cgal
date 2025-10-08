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
  // Compute vectors
  std::array<double, 3> vec1 = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  std::array<double, 3> vec2 = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};

  // Compute cross product
  std::array<double, 3> cross_product = {vec1[1] * vec2[2] - vec1[2] * vec2[1],
                                         vec1[2] * vec2[0] - vec1[0] * vec2[2],
                                         vec1[0] * vec2[1] - vec1[1] * vec2[0]};

  // Compute magnitude of cross product
  double magnitude = std::sqrt(cross_product[0] * cross_product[0] +
                               cross_product[1] * cross_product[1] +
                               cross_product[2] * cross_product[2]);

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
    QKey key{llround(p.x() * inv), llround(p.y() * inv), llround(p.z() * inv)};

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
    for (auto he : CGAL::halfedges_around_face(tm.halfedge(f), tm))
      tri[k++] = vertex_index_map[CGAL::target(he, tm)];

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