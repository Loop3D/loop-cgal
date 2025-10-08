#include "clip.h"
#include "meshutils.h"
#include "numpymesh.h"
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

namespace PMP = CGAL::Polygon_mesh_processing;
using face_descriptor = TriangleMesh::Face_index;

TriangleMesh load_mesh(NumpyMesh mesh, bool verbose) {
  TriangleMesh tm;
  auto vertices_buf = mesh.vertices.unchecked<2>();
  auto triangles_buf = mesh.triangles.unchecked<2>();
  std::vector<TriangleMesh::Vertex_index> vertex_indices;

  if (verbose) {
    std::cout << "Loading mesh with " << vertices_buf.shape(0)
              << " vertices and " << triangles_buf.shape(0) << " triangles."
              << std::endl;
  }

  // Assemble CGAL mesh objects from numpy/pybind11 arrays
  for (ssize_t i = 0; i < vertices_buf.shape(0); ++i) {
    vertex_indices.push_back(tm.add_vertex(
        Point(vertices_buf(i, 0), vertices_buf(i, 1), vertices_buf(i, 2))));
  }
  for (ssize_t i = 0; i < triangles_buf.shape(0); ++i) {
    tm.add_face(vertex_indices[triangles_buf(i, 0)],
                vertex_indices[triangles_buf(i, 1)],
                vertex_indices[triangles_buf(i, 2)]);
  }

  if (verbose) {
    std::cout << "Loaded mesh with " << tm.number_of_vertices()
              << " vertices and " << tm.number_of_faces() << " faces."
              << std::endl;
  }

  return tm;
}

Plane load_plane(NumpyPlane plane, bool verbose) {
  auto normal_buf = plane.normal.unchecked<1>();
  auto point_buf = plane.origin.unchecked<1>();
  if (verbose) {
    std::cout << "Loading plane with normal (" << normal_buf(0) << ", "
              << normal_buf(1) << ", " << normal_buf(2) << ") and point ("
              << point_buf(0) << ", " << point_buf(1) << ", " << point_buf(2)
              << ")." << std::endl;
  }
  return Plane(Point(point_buf(0), point_buf(1), point_buf(2)),
               Vector(normal_buf(0), normal_buf(1), normal_buf(2)));
}
// -----------------------------------------------------------------------------
//  Robust remesher that bails out on pathological micro‑patches
// -----------------------------------------------------------------------------
void refine_mesh(TriangleMesh &mesh, bool split_long_edges, bool verbose,
                 double target_edge_length, int number_of_iterations,
                 bool protect_constraints, bool relax_constraints) {
  // ------------------------------------------------------------------
  // 0.  Guard‑rail: sensible target length w.r.t. bbox
  // ------------------------------------------------------------------
  CGAL::Bbox_3 bb = PMP::bbox(mesh);
  const double bbox_diag = std::sqrt(CGAL::square(bb.xmax() - bb.xmin()) +
                                     CGAL::square(bb.ymax() - bb.ymin()) +
                                     CGAL::square(bb.zmax() - bb.zmin()));
  PMP::remove_isolated_vertices(mesh);
  if (target_edge_length < 1e-4 * bbox_diag) {
    if (verbose)
      std::cout << "  ! target_edge_length (" << target_edge_length
                << ") too small – skipping remesh\n";
    return;
  }

  // ------------------------------------------------------------------
  // 1.  Quick diagnostics
  // ------------------------------------------------------------------
  double min_e = std::numeric_limits<double>::max(), max_e = 0.0;
  for (auto e : mesh.edges()) {
    const double l = PMP::edge_length(e, mesh);
    min_e = std::min(min_e, l);
    max_e = std::max(max_e, l);
  }
  if (verbose)
    std::cout << "      edge length range: [" << min_e << ", " << max_e
              << "]  target = " << target_edge_length << '\n';

  if (!CGAL::is_valid_polygon_mesh(mesh, verbose) && verbose)
    std::cout << "      ! mesh is not a valid polygon mesh\n";

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
  const std::size_t n_faces = mesh.number_of_faces();
  if (n_faces < 40) {
    if (split_long_edges)
      PMP::split_long_edges(edges(mesh), target_edge_length, mesh);
    if (verbose)
      std::cout << "      → tiny patch (" << n_faces
                << " faces) – isotropic remesh skipped\n";
    return;
  }

  // ------------------------------------------------------------------
  // 4.  Normal isotropic remeshing loop
  // ------------------------------------------------------------------
  for (int iter = 0; iter < number_of_iterations; ++iter) {
    if (split_long_edges)
      PMP::split_long_edges(edges(mesh), target_edge_length, mesh);

    std::set<TriangleMesh::Edge_index> border_edges =
        collect_border_edges(mesh);

    PMP::isotropic_remeshing(
        faces(mesh), target_edge_length, mesh,
        CGAL::parameters::number_of_iterations(1) // one sub‑iteration per loop
            .edge_is_constrained_map(
                CGAL::make_boolean_property_map(border_edges))
            .protect_constraints(protect_constraints)
            .relax_constraints(relax_constraints));
  }

  if (verbose)
    std::cout << "Refined mesh → " << mesh.number_of_vertices() << " V, "
              << mesh.number_of_faces() << " F\n";
  if (!CGAL::is_valid_polygon_mesh(mesh, verbose) && verbose)
    std::cout << "      ! mesh is not a valid polygon mesh after remeshing\n";
}

bool plane_cuts_mesh(const TriangleMesh &mesh, const Plane &P) {
  bool has_pos = false, has_neg = false;

  for (auto v : mesh.vertices()) {
    const auto side = P.oriented_side(mesh.point(v));
    if (side == CGAL::ON_POSITIVE_SIDE)
      has_pos = true;
    else if (side == CGAL::ON_NEGATIVE_SIDE)
      has_neg = true;

    if (has_pos && has_neg)
      return true; // found both sides → cut
  }
  return false; // all vertices on one side
}

NumpyMesh clip_plane(NumpyMesh tm, NumpyPlane clipper,
                     double target_edge_length, bool remesh_before_clipping,
                     bool remesh_after_clipping, bool remove_degenerate_faces,
                     double duplicate_vertex_threshold, double area_threshold,
                     bool protect_constraints, bool relax_constraints,
                     bool verbose) {
  int number_of_iterations = 3; // Number of remeshing iterations
  if (verbose) {
    std::cout << "Starting clipping process." << std::endl;
    std::cout << "Loading data from NumpyMesh." << std::endl;
  }
  TriangleMesh _tm = load_mesh(tm, verbose);
  if (verbose) {
    std::cout << "Loaded mesh." << std::endl;
  }
  Plane _clipper = load_plane(clipper, verbose);
  if (verbose) {
    std::cout << "Loaded plane." << std::endl;
  }
  if (remesh_before_clipping) {
    if (verbose) {
      std::cout << "Remeshing before clipping." << std::endl;
    }
    refine_mesh(_tm, true, verbose, target_edge_length, number_of_iterations,
                protect_constraints, relax_constraints);
    // number_of_iterations);

    if (verbose) {
      std::cout << "Remeshing before clipping done." << std::endl;
    }
  }

  // make sure the meshes actually intersect. If they don't, just return mesh 1
  bool intersection = plane_cuts_mesh(_tm, _clipper);

  if (intersection) {
    // Clip tm with clipper
    if (verbose) {
      std::cout << "Clipping tm with clipper." << std::endl;
    }
    bool flag = PMP::clip(_tm, _clipper, CGAL::parameters::clip_volume(false));
    // PMP::triangulate_faces(_tm);
    if (verbose) {
      std::cout << "Clipping done." << std::endl;
    }
    if (!flag) {
      std::cerr << "Clipping failed." << std::endl;
      return {};
    } else {
      if (remesh_after_clipping) {

        if (verbose) {
          std::cout << "Remeshing after clipping." << std::endl;
        }
        if (verbose)
          std::cout << "  – stitching borders…" << std::endl;

        PMP::stitch_borders(_tm);
        if (verbose)
          std::cout << "  – merging dup vertices…" << std::endl;
        PMP::merge_duplicated_vertices_in_boundary_cycles(_tm);
        if (verbose)
          std::cout << "  – isotropic remeshing…" << std::endl;
        refine_mesh(_tm, true, verbose, target_edge_length,
                    number_of_iterations, protect_constraints,
                    relax_constraints);

        if (verbose) {
          std::cout << "Remeshing after clipping done." << std::endl;
        }
      }
      if (remove_degenerate_faces) {
        if (verbose) {
          std::cout << "Removing degenerate faces." << std::endl;
        }
        std::set<TriangleMesh::Edge_index> protected_edges =
            collect_border_edges(_tm);
#if CGAL_VERSION_NR >= 1060000000
        bool beautify_flag = PMP::remove_almost_degenerate_faces(
            faces(_tm), _tm,
            CGAL::parameters::edge_is_constrained_map(
                CGAL::make_boolean_property_map(protected_edges)));
#else
        bool beautify_flag = PMP::remove_degenerate_faces(
            faces(_tm), _tm,
            CGAL::parameters::edge_is_constrained_map(
                CGAL::make_boolean_property_map(protected_edges)));
#endif
        if (!beautify_flag) {
          std::cerr << "Removing degenerate faces failed." << std::endl;
        }
        if (verbose) {
          std::cout << "Removing degenerate faces done." << std::endl;
        }
      }
    }
  } else {
    if (verbose)
      std::cout << "Meshes do not intersect. Returning tm." << std::endl;
  }
  if (verbose) {
    std::cout << "Clipping done." << std::endl;
  }

  // store the result in a numpymesh object for sending back to Python

  NumpyMesh result =
      export_mesh(_tm, area_threshold, duplicate_vertex_threshold);
  if (verbose) {
    std::cout << "Exported clipped mesh with " << result.vertices.shape(0)
              << " vertices and " << result.triangles.shape(0) << " triangles."
              << std::endl;
  }
  return result;
}
NumpyMesh clip_surface(NumpyMesh tm, NumpyMesh clipper,
                       double target_edge_length, bool remesh_before_clipping,
                       bool remesh_after_clipping, bool remove_degenerate_faces,
                       double duplicate_vertex_threshold, double area_threshold,
                       bool protect_constraints, bool relax_constraints,
                       bool verbose) {
  if (verbose) {
    std::cout << "Starting clipping process." << std::endl;
    std::cout << "Loading data from NumpyMesh." << std::endl;
  }
  TriangleMesh _tm = load_mesh(tm, verbose);
  TriangleMesh _clipper = load_mesh(clipper, verbose);
  if (verbose) {
    std::cout << "Loaded meshes." << std::endl;
  }
  PMP::remove_isolated_vertices(_tm);
  PMP::remove_isolated_vertices(_clipper);
  if (!CGAL::is_valid_polygon_mesh(_tm, verbose)) {
    std::cerr << "tm is invalid!" << std::endl;
    if (verbose)
    {
      CGAL::is_valid_polygon_mesh(_tm, true);
    }
  }
  if (!CGAL::is_valid_polygon_mesh(_clipper, verbose)) {
    std::cerr << "clipper is invalid!" << std::endl;
  }
  // Parameters for isotropic remeshing
  const unsigned int number_of_iterations = 3; // Number of remeshing iterations
  if (remesh_before_clipping) {
    if (verbose) {
      std::cout << "Remeshing before clipping." << std::endl;
    }
    refine_mesh(_tm, true, verbose, target_edge_length, number_of_iterations,
                protect_constraints, relax_constraints);
    // refine_mesh(_clipper, true, verbose, target_edge_length,
    //             number_of_iterations, protect_constraints,
    //             relax_constraints);

    if (verbose) {
      std::cout << "Remeshing before clipping done." << std::endl;
    }
  }

  // make sure the meshes actually intersect. If they don't, just return mesh 1
  bool intersection = PMP::do_intersect(_tm, _clipper);
  if (intersection) {
    // Clip tm with clipper
    if (verbose) {
      std::cout << "Clipping tm with clipper." << std::endl;
    }
    bool flag = PMP::clip(_tm, _clipper);
    // PMP::triangulate_faces(_tm);
    if (verbose) {
      std::cout << "Clipping done." << std::endl;
    }
    if (!flag) {
      std::cerr << "Clipping failed." << std::endl;
      return {};
    } else {
      if (remesh_after_clipping) {

        if (verbose) {
          std::cout << "Remeshing after clipping." << std::endl;
        }
        if (verbose)
          std::cout << "  – stitching borders…" << std::endl;
        PMP::stitch_borders(_tm);
        if (verbose)
          std::cout << "  – merging dup vertices…" << std::endl;
        PMP::merge_duplicated_vertices_in_boundary_cycles(_tm);
        if (verbose)
          std::cout << "  – isotropic remeshing…" << std::endl;
        refine_mesh(_tm, true, verbose, target_edge_length,
                    number_of_iterations, protect_constraints,
                    relax_constraints);

        if (verbose) {
          std::cout << "Remeshing after clipping done." << std::endl;
        }
      }
      if (remove_degenerate_faces) {
        if (verbose) {
          std::cout << "Removing degenerate faces." << std::endl;
        }
        std::set<TriangleMesh::Edge_index> protected_edges =
            collect_border_edges(_tm);

#if CGAL_VERSION_NR >= 1060000000
        bool beautify_flag = PMP::remove_almost_degenerate_faces(
            faces(_tm), _tm,
            CGAL::parameters::edge_is_constrained_map(
                CGAL::make_boolean_property_map(protected_edges)));
#else
        bool beautify_flag = PMP::remove_degenerate_faces(
            faces(_tm), _tm,
            CGAL::parameters::edge_is_constrained_map(
                CGAL::make_boolean_property_map(protected_edges)));
#endif
        if (!beautify_flag) {
          std::cerr << "Removing degenerate faces failed." << std::endl;
        }
        if (verbose) {
          std::cout << "Removing degenerate faces done." << std::endl;
        }
      }
    }
  } else {
    if (verbose)
    {
      std::cout << "Meshes do not intersect. Returning tm." << std::endl;
    }
  }
  if (verbose) {
    std::cout << "Clipping done." << std::endl;
  }

  // store the result in a numpymesh object for sending back to Python

  NumpyMesh result =
      export_mesh(_tm, area_threshold, duplicate_vertex_threshold);
  if (verbose) {
    std::cout << "Exported clipped mesh with " << result.vertices.shape(0)
              << " vertices and " << result.triangles.shape(0) << " triangles."
              << std::endl;
  }
  return result;
}

std::vector<NumpyMesh>
corefine_mesh(NumpyMesh tm1, NumpyMesh tm2, double target_edge_length,
              double duplicate_vertex_threshold, double area_threshold,
              int number_of_iterations, bool relax_constraints,
              bool protect_constraints, bool verbose) {
  // Load the meshes
  TriangleMesh _tm1 = load_mesh(tm1, false);
  TriangleMesh _tm2 = load_mesh(tm2, false);
  PMP::split_long_edges(edges(_tm1), target_edge_length, _tm1);
  PMP::split_long_edges(edges(_tm2), target_edge_length, _tm2);

  // Perform corefinement
  PMP::corefine(_tm1, _tm2);
  // Find shared edges
  std::set<TriangleMesh::Edge_index> tm_1_shared_edges;
  std::set<TriangleMesh::Edge_index> tm_2_shared_edges;
  for (const auto &edge1 : _tm1.edges()) {
    Point p1 = _tm1.point(CGAL::source(edge1, _tm1));
    Point p2 = _tm1.point(CGAL::target(edge1, _tm1));

    for (const auto &edge2 : _tm2.edges()) {
      Point q1 = _tm2.point(CGAL::source(edge2, _tm2));
      Point q2 = _tm2.point(CGAL::target(edge2, _tm2));

      // Check if the edges are identical (considering both orientations)
      if ((p1 == q1 && p2 == q2) || (p1 == q2 && p2 == q1)) {
        tm_1_shared_edges.insert(edge1);
        tm_2_shared_edges.insert(edge2);
        break;
      }
    }
  }
  if (verbose)
  {
    std::cout << "Found " << tm_1_shared_edges.size()
              << " shared edges in tm1 and " << tm_2_shared_edges.size()
              << " shared edges in tm2." << std::endl;
  }
  // std::set<TriangleMesh::Edge_index> constrained_edges;

  std::set<TriangleMesh::Edge_index> boundary_edges =
      collect_border_edges(_tm1);
  std::set<TriangleMesh::Edge_index> boundary_edges2 =
      collect_border_edges(_tm2);

  tm_1_shared_edges.insert(boundary_edges.begin(), boundary_edges.end());
  tm_2_shared_edges.insert(boundary_edges2.begin(), boundary_edges2.end());
  // Refine the meshes
  // Perform isotropic remeshing on _tm
  PMP::isotropic_remeshing(
      faces(_tm1), // Range of faces to remesh
      target_edge_length, _tm1,
      CGAL::parameters::number_of_iterations(number_of_iterations)
          .edge_is_constrained_map(
              CGAL::make_boolean_property_map(tm_1_shared_edges))
          .relax_constraints(relax_constraints)
          .protect_constraints(protect_constraints));
  PMP::isotropic_remeshing(
      faces(_tm2), // Range of faces to remesh
      target_edge_length, _tm2,
      CGAL::parameters::number_of_iterations(number_of_iterations)
          .edge_is_constrained_map(
              CGAL::make_boolean_property_map(tm_2_shared_edges))
          .relax_constraints(relax_constraints)
          .protect_constraints(protect_constraints));

  if (verbose)
  {
    std::cout << "Corefinement done." << std::endl;
  }
  return {
      export_mesh(_tm1, area_threshold, duplicate_vertex_threshold),
      export_mesh(_tm2, area_threshold, duplicate_vertex_threshold)};
}
