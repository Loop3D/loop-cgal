#include "mesh.h"
#include "meshutils.h"
#include "globals.h"
#include "sizet.h"
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
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

std::vector<std::pair<Point, Point>>
TriMesh::snapshot_constraint_coords(bool interior_only) const
{
  std::vector<std::pair<Point, Point>> coords;
  coords.reserve(_fixedEdges.size());
  for (auto e : _fixedEdges)
  {
    // _fixedEdges may hold edges that clip() has since tombstoned (removed but
    // not yet garbage-collected); skip those so we never dereference a dead
    // halfedge or capture a stale position.
    if (!_mesh.has_valid_index(e) || _mesh.is_removed(e))
      continue;
    auto h = _mesh.halfedge(e);
    if (interior_only &&
        (_mesh.is_border(h) || _mesh.is_border(_mesh.opposite(h))))
      continue;
    coords.emplace_back(_mesh.point(_mesh.source(h)), _mesh.point(_mesh.target(h)));
  }
  return coords;
}

void TriMesh::rebuild_fixed_edges_from_coords(
    const std::vector<std::pair<Point, Point>> &saved)
{
  // Border edges of the (possibly newly built) mesh are always constraints —
  // this re-derives the outer boundary plus any fresh cut boundary left by the
  // clip, exactly as init() would.
  _fixedEdges = collect_border_edges(_mesh);

  // Coordinate → vertex lookup for the current mesh. Positions are compared by
  // exact double value: surviving vertices keep their coordinates bit-for-bit
  // across clip()+collect_garbage() and across the exact round trip (an exact
  // number built from a double converts back to that same double), so no
  // tolerance is needed to re-find them.
  std::map<std::tuple<double, double, double>, TriangleMesh::Vertex_index> lut;
  for (auto v : _mesh.vertices())
  {
    const auto &p = _mesh.point(v);
    lut.emplace(std::make_tuple(p.x(), p.y(), p.z()), v);
  }

  for (const auto &pr : saved)
  {
    auto it0 = lut.find(std::make_tuple(pr.first.x(), pr.first.y(), pr.first.z()));
    auto it1 = lut.find(std::make_tuple(pr.second.x(), pr.second.y(), pr.second.z()));
    if (it0 == lut.end() || it1 == lut.end())
      continue; // an endpoint no longer exists — constraint did not survive
    auto h = _mesh.halfedge(it0->second, it1->second);
    if (h == TriangleMesh::null_halfedge())
      continue; // endpoints exist but are no longer directly connected
    _fixedEdges.insert(_mesh.edge(h));
  }

  _edge_is_constrained_map = CGAL::make_boolean_property_map(_fixedEdges);
}

// Move constructor: _edge_is_constrained_map stores a raw pointer into
// _fixedEdges, so after relocating _fixedEdges we must re-bind the map.
TriMesh::TriMesh(TriMesh&& other) noexcept
    : _mesh(std::move(other._mesh))
    , _fixedEdges(std::move(other._fixedEdges))
    , _edge_is_constrained_map(CGAL::make_boolean_property_map(_fixedEdges))
{}

TriMesh& TriMesh::operator=(TriMesh&& other) noexcept
{
    if (this != &other) {
        _mesh = std::move(other._mesh);
        _fixedEdges = std::move(other._fixedEdges);
        _edge_is_constrained_map = CGAL::make_boolean_property_map(_fixedEdges);
    }
    return *this;
}

// Copy constructor: same issue — copy creates a new _fixedEdges at a new
// address, so the property map must be re-bound to *this* object's set.
TriMesh::TriMesh(const TriMesh& other)
    : _mesh(other._mesh)
    , _fixedEdges(other._fixedEdges)
    , _edge_is_constrained_map(CGAL::make_boolean_property_map(_fixedEdges))
{}

TriMesh& TriMesh::operator=(const TriMesh& other)
{
    if (this != &other) {
        _mesh = other._mesh;
        _fixedEdges = other._fixedEdges;
        _edge_is_constrained_map = CGAL::make_boolean_property_map(_fixedEdges);
    }
    return *this;
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
    TriangleMesh::Vertex_index v0 = TriangleMesh::Vertex_index(pairs_buf(i, 0));
    TriangleMesh::Vertex_index v1 = TriangleMesh::Vertex_index(pairs_buf(i, 1));
    if (!_mesh.is_valid(v0) || !_mesh.is_valid(v1))
    {
      std::cerr << "Invalid vertex indices: (" << v0 << ", " << v1 << ")"
                << std::endl;
      continue; // Skip invalid vertex pairs
    }
    TriangleMesh::Halfedge_index edge = _mesh.halfedge(v0, v1);
    if (edge == TriangleMesh::null_halfedge())
    {
      std::cerr << "Half-edge is null for vertices (" << v0 << ", " << v1 << ")"
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

int TriMesh::cutWithSurface(TriMesh &clipper,
                             bool preserve_intersection,
                             bool preserve_intersection_clipper,
                            bool use_exact_kernel,
                            bool extend)
{
  if (LoopCGAL::verbose)
  {
    std::cout << "Cutting mesh with surface." << std::endl;
  }

  // Validate input meshes. A bad or empty input is a programming error, not a
  // legitimate no-op, so raise rather than return 0 (which is reserved for the
  // meshes-don't-intersect case below). pybind11 maps invalid_argument ->
  // Python ValueError.
  if (_mesh.number_of_vertices() == 0 || _mesh.number_of_faces() == 0)
    throw std::invalid_argument("cutWithSurface: source mesh is empty");

  if (clipper._mesh.number_of_vertices() == 0 ||
      clipper._mesh.number_of_faces() == 0)
    throw std::invalid_argument("cutWithSurface: clipper mesh is empty");

  if (!CGAL::is_valid_polygon_mesh(_mesh, LoopCGAL::verbose))
    throw std::invalid_argument("cutWithSurface: source mesh is not a valid polygon mesh");

  if (!CGAL::is_valid_polygon_mesh(clipper._mesh, LoopCGAL::verbose))
    throw std::invalid_argument("cutWithSurface: clipper mesh is not a valid polygon mesh");

  // Merge any collocated border vertices on the target before clipping.
  // Collocated vertices produce zero-area faces whose degenerate bounding boxes
  // can make PMP::do_intersect return false even when the meshes overlap.
  PMP::stitch_borders(_mesh);
  // Remove any isolated (orphan) vertices — they survive PMP::clip unchanged
  // and would pollute the output point set with stale positions.
  PMP::remove_isolated_vertices(_mesh);

  // -----------------------------------------------------------------------
  // Grow the clipper by extruding a skirt of new triangles outward from each
  // boundary edge.  Each skirt quad's normal matches its adjacent border face,
  // so the extension is along the surface tangent — not a global scale.
  // Interior vertices and faces are completely untouched, so the cut location
  // is preserved exactly for both planar and curved (listric) clippers.
  // -----------------------------------------------------------------------
  // Copy the clipper and stitch any collocated border vertices first.
  // stitch_borders can remove/merge vertices, so all subsequent passes must
  // operate on extended_mesh (not clipper._mesh) to keep vertex descriptors valid.
  TriangleMesh extended_mesh = clipper._mesh;
  PMP::stitch_borders(extended_mesh);

  // Grow the clipper by extruding a skirt of new triangles outward from each
  // boundary edge, so a clipper smaller than the target still cuts all the way
  // through. Skipped when extend=false: callers passing an already-domain-
  // spanning clipper must NOT be
  // re-extended — the skirt of a large surface folds on concave boundaries and
  // corefinement then rejects the near-coplanar overlap.
  if (extend)
  {
    CGAL::Bbox_3 target_bb = PMP::bbox(_mesh);
    const double target_diag = std::sqrt(
        CGAL::square(target_bb.xmax() - target_bb.xmin()) +
        CGAL::square(target_bb.ymax() - target_bb.ymin()) +
        CGAL::square(target_bb.zmax() - target_bb.zmin()));

    // Pass 1: accumulate per-vertex outward directions from each border halfedge
    // in extended_mesh (post-stitch).  For a border halfedge h (source→target),
    // the outward direction is fn × d, where fn is the adjacent face normal and
    // d is the normalised edge direction.  This lies in the face's tangent plane
    // and points away from the interior.
    std::map<TriangleMesh::Vertex_index, Vector> outward_sum;
    for (auto he : extended_mesh.halfedges())
    {
      if (!extended_mesh.is_border(he)) continue;

      const Point &ps = extended_mesh.point(extended_mesh.source(he));
      const Point &pt = extended_mesh.point(extended_mesh.target(he));
      Vector d = pt - ps;
      const double d_len = std::sqrt(d.squared_length());
      if (d_len < 1e-10) continue;
      d = d / d_len;

      const auto adj_face = extended_mesh.face(extended_mesh.opposite(he));
      const Vector fn = PMP::compute_face_normal(adj_face, extended_mesh);
      const Vector out = CGAL::cross_product(fn, d);
      const double out_len = std::sqrt(out.squared_length());
      if (out_len < 1e-10) continue;

      auto vs = extended_mesh.source(he);
      auto vt = extended_mesh.target(he);
      outward_sum.try_emplace(vs, 0.0, 0.0, 0.0);
      outward_sum.try_emplace(vt, 0.0, 0.0, 0.0);
      outward_sum[vs] = outward_sum[vs] + out / out_len;
      outward_sum[vt] = outward_sum[vt] + out / out_len;
    }

    // Pass 2: add one new vertex per boundary vertex pushed outward by target_diag,
    // and stitch a skirt quad (two triangles) per boundary edge of extended_mesh.
    // The winding order (source, target, target_new) produces normals consistent
    // with the adjacent interior face.
    std::map<TriangleMesh::Vertex_index, TriangleMesh::Vertex_index> skirt_vertex;
    for (auto &[v, dir_sum] : outward_sum)
    {
      const double len = std::sqrt(dir_sum.squared_length());
      if (len < 1e-10) continue;
      const Vector dir = dir_sum / len;
      const Point &p = extended_mesh.point(v);
      skirt_vertex[v] = extended_mesh.add_vertex(
          Point(p.x() + target_diag * dir.x(),
                p.y() + target_diag * dir.y(),
                p.z() + target_diag * dir.z()));
    }

    for (auto he : extended_mesh.halfedges())
    {
      if (!extended_mesh.is_border(he)) continue;
      auto vs = extended_mesh.source(he);
      auto vt = extended_mesh.target(he);
      if (!skirt_vertex.count(vs) || !skirt_vertex.count(vt)) continue;
      auto vs_new = skirt_vertex[vs];
      auto vt_new = skirt_vertex[vt];
      extended_mesh.add_face(vs, vt, vt_new);
      extended_mesh.add_face(vs, vt_new, vs_new);
    }

    if (LoopCGAL::verbose)
      std::cout << "  cutWithSurface: added skirt of "
                << skirt_vertex.size() << " new vertices over target_diag="
                << target_diag << "\n";
  }

  TriMesh scaled_clipper(std::move(extended_mesh));

  const int faces_before = static_cast<int>(_mesh.number_of_faces());

  bool intersection = PMP::do_intersect(_mesh, scaled_clipper._mesh);
  if (intersection)
  {
    // Clip tm with clipper
    if (LoopCGAL::verbose)
    {
      std::cout << "Clipping tm with clipper." << std::endl;
    }

    bool flag = false;
    try
    {
      if (use_exact_kernel){
        // The exact round trip rebuilds _mesh from scratch (convert_to_double_mesh),
        // so every Edge_index in _fixedEdges is invalidated and the constrained
        // map cannot be threaded through the exact clip. Capture constraints by
        // geometry beforehand and re-resolve them against the rebuilt mesh after.
        auto saved_constraints = snapshot_constraint_coords();
        Exact_Mesh exact_clipper = convert_to_exact(scaled_clipper);
        Exact_Mesh exact_mesh = convert_to_exact(*this);
        flag = PMP::clip(exact_mesh, exact_clipper, CGAL::parameters::clip_volume(false));
        set_mesh(convert_to_double_mesh(exact_mesh));
        if (flag)
          rebuild_fixed_edges_from_coords(saved_constraints);
      }
      else{
        // Pass the constrained-edge map for both meshes: CGAL reads existing
        // constraints on input and, on output, marks the intersection edges and
        // any surviving/split constraint edges. After the clip, _fixedEdges holds
        // valid (pre-garbage-collection) indices, so snapshot it by geometry,
        // collect garbage, then re-resolve — Edge_index values do not survive
        // collect_garbage().
        flag = PMP::clip(
            _mesh, scaled_clipper._mesh,
            CGAL::parameters::edge_is_constrained_map(_edge_is_constrained_map)
                .clip_volume(false),
            CGAL::parameters::edge_is_constrained_map(
                scaled_clipper._edge_is_constrained_map));
        if (flag)
        {
          auto saved_constraints = snapshot_constraint_coords();
          if (_mesh.has_garbage())
            _mesh.collect_garbage();
          rebuild_fixed_edges_from_coords(saved_constraints);
        }
      }
    }
    catch (const std::exception &e)
    {
      // The meshes intersect, so the clip was expected to succeed — a thrown
      // exception here is a real failure, not a no-op. Rethrow with context
      // (pybind11 maps runtime_error -> Python RuntimeError).
      throw std::runtime_error(std::string("cutWithSurface: clip failed: ") + e.what());
    }
    // The meshes intersect but PMP::clip reported failure — surface it rather
    // than silently returning 0 (which callers read as "no intersection").
    if (!flag)
      throw std::runtime_error("cutWithSurface: clip of intersecting meshes failed");
    if (LoopCGAL::verbose)
    {
      std::cout << "Clipping successful. Result has "
                << _mesh.number_of_vertices() << " vertices and "
                << _mesh.number_of_faces() << " faces." << std::endl;
    }
  }
  else
  {
    // Legitimate no-op: nothing to cut. Return 0 (see faces_before-faces_after).
    if (LoopCGAL::verbose)
    {
      std::cout << "Meshes do not intersect. No clipping performed."
                << std::endl;
    }
  }

  const int faces_after = static_cast<int>(_mesh.number_of_faces());
  if (faces_after >= faces_before && LoopCGAL::verbose)
  {
    std::cerr << "Warning: cutWithSurface removed no faces (before="
              << faces_before << ", after=" << faces_after
              << "). Clipper may not extend beyond the mesh." << std::endl;
  }
  return faces_before - faces_after;
}

int TriMesh::corefine(TriMesh &other, bool use_exact_kernel)
{
  // The inexact kernel path is unsafe by construction: TriangleMesh uses
  // Simple_cartesian<double>, whose predicates are inexact, and CGAL
  // corefinement requires exact predicates to make consistent branching
  // decisions. On Simple_cartesian it does not merely give a wrong result — it
  // hard-crashes (SIGBUS) even on trivial valid inputs, which no try/catch can
  // trap. Refuse it up front with a catchable error instead. The exact path
  // converts to the EPECK Exact_Mesh, which is safe.
  if (!use_exact_kernel)
    throw std::invalid_argument(
        "corefine requires the exact kernel; use_exact_kernel=False is unsafe "
        "with the inexact predicate kernel and crashes. Call with "
        "use_exact_kernel=True (the default).");

  if (_mesh.number_of_faces() == 0 || other._mesh.number_of_faces() == 0)
    throw std::invalid_argument("corefine: called on an empty mesh");

  // Merge collocated border vertices first — corefinement needs valid,
  // duplicate-free borders to compute a clean intersection polyline.
  PMP::stitch_borders(_mesh);
  PMP::remove_isolated_vertices(_mesh);
  PMP::stitch_borders(other._mesh);
  PMP::remove_isolated_vertices(other._mesh);

  const std::size_t v_before = _mesh.number_of_vertices();

  try
  {
    Exact_Mesh em1 = convert_to_exact(*this);
    Exact_Mesh em2 = convert_to_exact(other);
    PMP::corefine(em1, em2);
    set_mesh(convert_to_double_mesh(em1));
    other.set_mesh(convert_to_double_mesh(em2));
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error(std::string("corefine failed: ") + e.what());
  }

  if (_mesh.has_garbage())
    _mesh.collect_garbage();
  if (other._mesh.has_garbage())
    other._mesh.collect_garbage();

  // Re-derive border edges (the intersection split changed topology on both).
  init();
  other.init();

  const std::size_t v_after = _mesh.number_of_vertices();
  if (LoopCGAL::verbose)
  {
    std::cout << "corefine: this mesh " << v_before << " -> " << v_after
              << " vertices." << std::endl;
  }
  return static_cast<int>(v_after) - static_cast<int>(v_before);
}

int TriMesh::clipWithPlane(double a, double b, double c, double d, bool use_exact_kernel)
{
  if (_mesh.number_of_vertices() == 0 || _mesh.number_of_faces() == 0)
    throw std::invalid_argument("clipWithPlane: source mesh is empty");

  // PMP::clip is not idempotent. Re-clipping a mesh that already lies entirely
  // inside the halfspace removes no face, but still corefines the mesh against
  // the plane and splits every edge along the existing seam. Repeating a plane
  // therefore grows the mesh without bound -- and each exact round trip rounds
  // the new seam points onto coincident doubles, which is what eventually takes
  // PMP::clip down (Loop3D/loop-cgal#14). Nothing strictly outside the halfspace
  // means nothing to remove, so return the documented no-op untouched.
  const double normal_length = std::sqrt(a * a + b * b + c * c);
  if (!(normal_length > 0.0))
    throw std::invalid_argument("clipWithPlane: degenerate plane normal (a=b=c=0)");
  double max_coord = 0.0;
  double max_signed_distance = -std::numeric_limits<double>::infinity();
  for (auto v : _mesh.vertices())
  {
    const auto &p = _mesh.point(v);
    max_coord = std::max(max_coord, std::abs(p.x()));
    max_coord = std::max(max_coord, std::abs(p.y()));
    max_coord = std::max(max_coord, std::abs(p.z()));
    const double signed_distance =
        (a * p.x() + b * p.y() + c * p.z() + d) / normal_length;
    max_signed_distance = std::max(max_signed_distance, signed_distance);
  }
  // Scale the tolerance to the mesh's own coordinate magnitude: the round trip
  // displaces a seam vertex by a few ulps of that magnitude, never more.
  const double on_plane_tol = 1e-12 * std::max(1.0, max_coord);
  if (max_signed_distance <= on_plane_tol)
  {
    if (LoopCGAL::verbose)
      std::cout << "clipWithPlane: mesh already inside the halfspace (max signed "
                << "distance " << max_signed_distance << " <= " << on_plane_tol
                << "), nothing to clip." << std::endl;
    return 0;
  }

  const int faces_before = static_cast<int>(_mesh.number_of_faces());

  // Capture constraints by geometry before the clip. The plane overload of
  // PMP::clip does not accept an edge_is_constrained_map (it uses an internal
  // one), and both the exact round trip and collect_garbage() below invalidate
  // every Edge_index, so _fixedEdges must be re-resolved by coordinate after.
  auto saved_constraints = snapshot_constraint_coords();

  try
  {
    if (use_exact_kernel)
    {
      Exact_K::Plane_3 plane(a, b, c, d);
      Exact_Mesh exact = convert_to_exact(*this);
      PMP::clip(exact, plane, CGAL::parameters::clip_volume(false));
      set_mesh(convert_to_double_mesh(exact));
    }
    else
    {
      Plane plane(a, b, c, d);
      PMP::clip(_mesh, plane, CGAL::parameters::clip_volume(false));
    }
  }
  catch (const std::exception &e)
  {
    // pybind11 maps runtime_error -> Python RuntimeError.
    throw std::runtime_error(std::string("clipWithPlane: clip failed: ") + e.what());
  }

  const int faces_after = static_cast<int>(_mesh.number_of_faces());
  if (LoopCGAL::verbose && faces_after >= faces_before)
  {
    std::cerr << "Warning: clipWithPlane removed no faces (before="
              << faces_before << ", after=" << faces_after << ")." << std::endl;
  }
  // Compact the mesh: PMP::clip marks removed elements as tombstones without
  // freeing them.  Iterating _mesh.vertices() when has_garbage()==true requires
  // scanning every tombstoned slot, so a mesh that has been clipped 6+ times
  // can accumulate enough tombstones to make get_points() take seconds.
  if (_mesh.has_garbage())
    _mesh.collect_garbage();

  // Re-derive the constraint set: border edges of the clipped mesh (including
  // the new cut boundary) plus any user-added interior constraints that
  // survived, re-resolved by coordinate. Without this, stale Edge_index values
  // in _fixedEdges would mark unrelated (recycled) edges as protected during a
  // later remesh().
  //
  // Limitation: a constraint that the plane cuts *through* is not recovered.
  // rebuild_fixed_edges_from_coords only re-resolves a constraint when its two
  // original endpoints are still directly connected; the plane overload of
  // PMP::clip takes no edge_is_constrained_map, so a split introduces a midpoint
  // that breaks that adjacency and the sub-edges are left unconstrained.
  rebuild_fixed_edges_from_coords(saved_constraints);
  return faces_before - faces_after;
}

NumpyMesh TriMesh::save(double area_threshold,
                        double duplicate_vertex_threshold)
{
  return export_mesh(_mesh, area_threshold, duplicate_vertex_threshold);
}

void TriMesh::cut_with_implicit_function(const std::vector<double> &property, double value, ImplicitCutMode cutmode, double snap_tol)
{
  if (LoopCGAL::verbose)
  {
    std::cout << "Cutting mesh with implicit function at value " << value << std::endl;
    std::cout << "Mesh has " << _mesh.number_of_vertices() << " vertices and "
              << _mesh.number_of_faces() << " faces." << std::endl;
    std::cout << "Property size: " << property.size() << std::endl;
  }
  // Compact first: vertex_properties/property are indexed by vertex.idx(), and a
  // mesh carrying tombstoned vertices (garbage from a prior clip) has idx() values
  // that exceed number_of_vertices() — indexing the property arrays with them
  // would read/write out of bounds. After collect_garbage() live vertices are
  // renumbered 0..n-1 contiguously, so idx() is always in range.
  if (_mesh.has_garbage())
    _mesh.collect_garbage();
  // The cut replaces _mesh wholesale, so the constraint set has to be re-derived
  // afterwards (see the end of this function).  Only user-added interior
  // constraints need carrying across by coordinate; borders are re-derived.
  auto saved_constraints = snapshot_constraint_coords(/*interior_only=*/true);
  if (property.size() != _mesh.number_of_vertices())
    throw std::invalid_argument(
        "cut_with_implicit_function: property size (" +
        std::to_string(property.size()) + ") does not match the number of "
        "mesh vertices (" + std::to_string(_mesh.number_of_vertices()) + ")");
  // Create a property map for vertex properties
  typedef boost::property_map<TriangleMesh, boost::vertex_index_t>::type VertexIndexMap;
  VertexIndexMap vim = get(boost::vertex_index, _mesh);
  std::vector<double> vertex_properties(_mesh.number_of_vertices());
  for (auto v : _mesh.vertices())
  {
    vertex_properties[vim[v]] = property[vim[v]];
  }

  // The sign-based case table below uses strict >/< comparisons, so a vertex
  // whose property equals `value` exactly matches neither side: its triangle is
  // left un-split and the isocontour is torn where it should pass through that
  // vertex.  Nudge any exactly-on-value property to the next representable
  // double (consistently to the positive side) so every vertex is unambiguously
  // classified; NaN (off-extent) compares false and is left untouched.  The
  // shift is sub-ULP, so it does not move the seam for any non-degenerate input.
  for (auto &vp : vertex_properties)
    if (vp == value)
      vp = std::nextafter(value, HUGE_VAL);
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
  // Which slots of each triangle are seam crossings rather than original
  // vertices.  A crossing may reuse an EXISTING vertex index (see snapping
  // below), so the index alone cannot tell the two apart, and the keep/discard
  // filter needs to know: a crossing lies on the isosurface by construction and
  // says nothing about which side the sub-triangle belongs to.
  std::vector<std::array<bool, 3>> newtris_on_seam(newtris.size(), {false, false, false});
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
    // for each edge of tri, check if edge crosses
    for (auto eid : tri2edge[t])
    {
      // A crossed edge is shared by two straddling triangles, so it is visited
      // twice.  Without this guard the second visit pushes a second, coincident
      // crossing vertex and overwrites new_point_on_edge[eid] with it, leaving
      // the two triangles referencing different vertices at the same position:
      // the mesh is torn along every seam, and the tear compounds over
      // sequential cuts.  The crossing is a function of the edge alone, so
      // reusing the vertex recorded by the first visit is exact.
      if (new_point_on_edge.count(eid))
        continue;
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

      // Snap the crossing to an existing endpoint when it lands within snap_tol
      // (measured as a fraction of the edge) of that endpoint.  Otherwise the
      // crossing point sits an infinitesimal distance from a real vertex, and
      // the sub-triangles built from it in the case table below are slivers that
      // a downstream pass would have to collapse.  Snapping reuses the existing
      // vertex instead: any sub-triangle that collapses to a repeated index is
      // dropped by the degeneracy check when the new mesh is assembled, so no
      // sliver is ever created.  newverts starts as a copy of verts, so the
      // original vertex indices ends.first/ends.second are valid indices into it.
      // Snapping declares that existing vertex to BE the crossing on THIS edge.
      // That is a per-edge fact, not a per-vertex one: the same vertex is still
      // strictly on its own side of the isosurface as far as every other
      // triangle around it is concerned.  Recording it in newtris_on_seam (per
      // slot) rather than by overwriting newvals (per vertex) keeps the two
      // apart — the latter leaks the assertion into every incident sub-triangle
      // and deletes kept surface wherever the seam runs through a vertex.
      if (ratio <= snap_tol)
      {
        new_point_on_edge[eid] = ends.first;
        continue;
      }
      if (ratio >= 1.0 - snap_tol)
      {
        new_point_on_edge[eid] = ends.second;
        continue;
      }

      Point p0 = verts[ends.first];
      Point p1 = verts[ends.second];
      Point np = Point(p0.x() + ratio * (p1.x() - p0.x()), p0.y() + ratio * (p1.y() - p0.y()), p0.z() + ratio * (p1.z() - p0.z()));
      newverts.push_back(np);
      newvals.push_back(value);
      new_point_on_edge[eid] = newverts.size() - 1;
    }

    // Edge ids of the three triangle sides, and the crossing vertex inserted on
    // each (SIZE_MAX where that side does not cross the isovalue).
    std::size_t e01 = edge_index_map[std::make_pair(std::min(tri[0], tri[1]), std::max(tri[0], tri[1]))];
    std::size_t e12 = edge_index_map[std::make_pair(std::min(tri[1], tri[2]), std::max(tri[1], tri[2]))];
    std::size_t e20 = edge_index_map[std::make_pair(std::min(tri[2], tri[0]), std::max(tri[2], tri[0]))];
    const std::size_t c01 = new_point_on_edge.count(e01) ? new_point_on_edge[e01] : SIZE_MAX;
    const std::size_t c12 = new_point_on_edge.count(e12) ? new_point_on_edge[e12] : SIZE_MAX;
    const std::size_t c20 = new_point_on_edge.count(e20) ? new_point_on_edge[e20] : SIZE_MAX;

    // Split the straddling triangle along the seam.  Exactly one vertex (the
    // "lone" vertex) lies on the opposite side of the isovalue from the other
    // two — the nudge above guarantees no vertex is exactly on it — so the two
    // triangle sides incident to the lone vertex are the ones that cross.  The
    // three sub-triangles produced depend only on *which* vertex is lone, not on
    // its sign; the sign only decides, in the assembly filter below, which
    // sub-triangles survive.  This collapses the six near-identical hand-written
    // cases (whose sign-paired variants produced identical oriented triangles)
    // into one table keyed on the lone vertex.
    const bool p0 = vertex_properties[tri[0]] > value;
    const bool p1 = vertex_properties[tri[1]] > value;
    const bool p2 = vertex_properties[tri[2]] > value;
    int lone;
    if (p1 == p2)      lone = 0; // tri[0] differs from tri[1] == tri[2]
    else if (p0 == p2) lone = 1; // tri[1] differs
    else               lone = 2; // tri[2] differs

    std::array<std::array<std::size_t, 3>, 3> sub;
    if (lone == 0)
      sub = {{{tri[0], c01, c20}, {c01, tri[1], tri[2]}, {c20, c01, tri[2]}}};
    else if (lone == 1)
      sub = {{{tri[0], c01, tri[2]}, {c01, c12, tri[2]}, {c01, tri[1], c12}}};
    else
      sub = {{{tri[0], tri[1], c12}, {tri[0], c12, c20}, {c20, c12, tri[2]}}};

    // Slot-for-slot with `sub` above: true where the entry is c01/c12/c20.
    std::array<std::array<bool, 3>, 3> sub_on_seam;
    if (lone == 0)
      sub_on_seam = {{{false, true, true}, {true, false, false}, {true, true, false}}};
    else if (lone == 1)
      sub_on_seam = {{{false, true, false}, {true, true, false}, {true, false, true}}};
    else
      sub_on_seam = {{{false, false, true}, {false, true, true}, {true, true, false}}};

    newtris[t] = sub[0];
    newtris_on_seam[t] = sub_on_seam[0];
    newtris.push_back(sub[1]);
    newtris_on_seam.push_back(sub_on_seam[1]);
    newtris.push_back(sub[2]);
    newtris_on_seam.push_back(sub_on_seam[2]);
  }

  // Build new CGAL mesh from newverts and newtris
  TriangleMesh newmesh;
  std::vector<TriangleMesh::Vertex_index> new_vhandles;
  new_vhandles.reserve(newverts.size());
  for (auto &p : newverts)
    new_vhandles.push_back(newmesh.add_vertex(p));
  for (std::size_t ti = 0; ti < newtris.size(); ++ti)
  {
    const auto &tri = newtris[ti];
    const auto &on_seam = newtris_on_seam[ti];
    // skip degenerate
    if (tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2])
      continue;

    // A sub-triangle belongs to the side its ORIGINAL (non-crossing) vertices
    // lie on; by construction they all lie on the same side.  Crossing vertices
    // sit on the isosurface and carry no side information, so they are skipped
    // rather than compared — which is what stops a wrong-side seam fringe (two
    // crossings plus one strictly-wrong-side vertex) from surviving, and equally
    // stops a legitimate keep-side sliver from being discarded.
    // A NaN value means the implicit function was not evaluated there (off
    // extent); such triangles are kept, as before.
    // Only the one-sided modes discard anything; PRESERVE_INTERSECTION splits
    // along the seam but keeps the whole surface.
    if (ImplicitCutMode::KEEP_POSITIVE_SIDE == cutmode ||
        ImplicitCutMode::KEEP_NEGATIVE_SIDE == cutmode)
    {
      const bool keep_positive = (ImplicitCutMode::KEEP_POSITIVE_SIDE == cutmode);
      bool on_keep_side = false;
      bool undetermined = false;
      for (int k = 0; k < 3; ++k)
      {
        if (on_seam[k])
          continue;
        const double v = newvals[tri[k]];
        if (std::isnan(v))
          undetermined = true;
        else if (keep_positive ? (v > value) : (v < value))
          on_keep_side = true;
      }
      if (!on_keep_side && !undetermined)
        continue;
    }

    newmesh.add_face(new_vhandles[tri[0]], new_vhandles[tri[1]], new_vhandles[tri[2]]);
  }

  // newverts carries every original vertex, including those used only by
  // dropped triangles.  Left in place they accumulate over sequential cuts, and
  // the caller — which must size its property array to n_vertices — pays to
  // evaluate its implicit function on all of them.
  PMP::remove_isolated_vertices(newmesh);
  if (newmesh.has_garbage())
    newmesh.collect_garbage();

  // Replace internal mesh
  _mesh = std::move(newmesh);

  // Re-derive the constraint set exactly as clipWithPlane does: border edges of
  // the new mesh (including the fresh cut boundary) plus any user-added interior
  // constraints that survived, re-resolved by coordinate.  Without this,
  // _fixedEdges keeps Edge_index values into the destroyed mesh, which alias
  // unrelated edges of the new one and mark them protected during a later
  // remesh(), and the cut boundary is never protected at all.
  //
  // Limitation (as for clipWithPlane): a constraint the seam passes *through* is
  // not recovered, because the inserted crossing vertex breaks the adjacency of
  // its original endpoints.
  //
  // With no interior constraints to carry across (the usual case) init() gives
  // the same result and skips building a coordinate lookup over every vertex,
  // which costs ~5 ms per cut on a 100k-vertex mesh.
  if (saved_constraints.empty())
    init();
  else
    rebuild_fixed_edges_from_coords(saved_constraints);
}

double TriMesh::area() const
{
  return PMP::area(_mesh);
}

std::size_t TriMesh::n_faces() const
{
  return _mesh.number_of_faces();
}

std::size_t TriMesh::n_vertices() const
{
  return _mesh.number_of_vertices();
}

pybind11::array_t<double> TriMesh::get_points() const
{
  std::size_t n = _mesh.number_of_vertices();
  pybind11::array_t<double> result({n, std::size_t(3)});
  auto r = result.mutable_unchecked<2>();
  std::size_t i = 0;
  for (auto v : _mesh.vertices()) {
    const auto& p = _mesh.point(v);
    r(i, 0) = p.x();
    r(i, 1) = p.y();
    r(i, 2) = p.z();
    ++i;
  }
  return result;
}

TriMesh TriMesh::clone() const
{
  TriMesh result(_mesh);  // copies _mesh via CGAL Surface_mesh copy-ctor, then init() sets border edges
  // Overwrite with the full fixed-edge set (border edges + any user-added edges).
  // CGAL Surface_mesh indices are integer handles and remain valid across copies of the same topology.
  result._fixedEdges = _fixedEdges;
  result._edge_is_constrained_map = CGAL::make_boolean_property_map(result._fixedEdges);
  return result;
}

// ---------------------------------------------------------------------------
// Native binary file I/O — avoids the pyvista round-trip for temp files.
//
// Format (little-endian, no padding):
//   [6 bytes]  magic "LCMESH"
//   [4 bytes]  n_vertices  (uint32)
//   [4 bytes]  n_faces     (uint32)
//   [nv×24]    vertices    (3 × float64 each: x, y, z)
//   [nf×12]    faces       (3 × uint32  each: i0, i1, i2)
// ---------------------------------------------------------------------------
void TriMesh::write_to_file(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("TriMesh::write_to_file — cannot open: " + path);

    // Build a compact vertex index map (mesh may have tombstoned entries
    // left by CGAL operations that have not been garbage-collected yet).
    std::vector<TriangleMesh::Vertex_index> valid_verts;
    std::unordered_map<uint32_t, uint32_t> vmap;
    valid_verts.reserve(_mesh.number_of_vertices());
    vmap.reserve(_mesh.number_of_vertices());
    for (auto v : _mesh.vertices()) {
        vmap[static_cast<uint32_t>(v.idx())] =
            static_cast<uint32_t>(valid_verts.size());
        valid_verts.push_back(v);
    }

    uint32_t nv = static_cast<uint32_t>(valid_verts.size());
    uint32_t nf = static_cast<uint32_t>(_mesh.number_of_faces());

    out.write("LCMESH", 6);
    out.write(reinterpret_cast<const char*>(&nv), 4);
    out.write(reinterpret_cast<const char*>(&nf), 4);

    for (auto v : valid_verts) {
        const auto& p = _mesh.point(v);
        double x = p.x(), y = p.y(), z = p.z();
        out.write(reinterpret_cast<const char*>(&x), 8);
        out.write(reinterpret_cast<const char*>(&y), 8);
        out.write(reinterpret_cast<const char*>(&z), 8);
    }

    for (auto f : _mesh.faces()) {
        auto h = _mesh.halfedge(f);
        for (int i = 0; i < 3; ++i) {
            uint32_t idx = vmap.at(static_cast<uint32_t>(_mesh.target(h).idx()));
            out.write(reinterpret_cast<const char*>(&idx), 4);
            h = _mesh.next(h);
        }
    }
}

TriMesh TriMesh::read_from_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("TriMesh::read_from_file — cannot open: " + path);

    char magic[6];
    in.read(magic, 6);
    if (!in || std::string(magic, 6) != "LCMESH")
        throw std::runtime_error("TriMesh::read_from_file — bad magic in: " + path);

    uint32_t nv, nf;
    in.read(reinterpret_cast<char*>(&nv), 4);
    in.read(reinterpret_cast<char*>(&nf), 4);
    if (!in)
        throw std::runtime_error("TriMesh::read_from_file — truncated header in: " + path);

    TriangleMesh mesh;
    std::vector<TriangleMesh::Vertex_index> verts(nv);
    for (uint32_t i = 0; i < nv; ++i) {
        double x, y, z;
        in.read(reinterpret_cast<char*>(&x), 8);
        in.read(reinterpret_cast<char*>(&y), 8);
        in.read(reinterpret_cast<char*>(&z), 8);
        if (!in)
            throw std::runtime_error("TriMesh::read_from_file — truncated vertex data in: " + path);
        verts[i] = mesh.add_vertex(Point(x, y, z));
    }

    for (uint32_t i = 0; i < nf; ++i) {
        uint32_t i0, i1, i2;
        in.read(reinterpret_cast<char*>(&i0), 4);
        in.read(reinterpret_cast<char*>(&i1), 4);
        in.read(reinterpret_cast<char*>(&i2), 4);
        if (!in)
            throw std::runtime_error("TriMesh::read_from_file — truncated face data in: " + path);
        if (i0 >= nv || i1 >= nv || i2 >= nv)
            throw std::runtime_error("TriMesh::read_from_file — face index out of range in: " + path);
        mesh.add_face(verts[i0], verts[i1], verts[i2]);
    }

    return TriMesh(std::move(mesh));   // private ctor calls init() → border edges
}

bool TriMesh::overlaps(const TriMesh& other, double bbox_tol) const
{
  CGAL::Bbox_3 b1 = PMP::bbox(_mesh);
  CGAL::Bbox_3 b2 = PMP::bbox(other._mesh);
  if (b1.xmax() < b2.xmin() - bbox_tol || b2.xmax() < b1.xmin() - bbox_tol ||
      b1.ymax() < b2.ymin() - bbox_tol || b2.ymax() < b1.ymin() - bbox_tol ||
      b1.zmax() < b2.zmin() - bbox_tol || b2.zmax() < b1.zmin() - bbox_tol) {
    return false;
  }
  return PMP::do_intersect(_mesh, other._mesh);
}

// Does this mesh intersect itself?
bool TriMesh::does_self_intersect() const
{
  return PMP::does_self_intersect(_mesh);
}
