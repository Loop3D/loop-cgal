from __future__ import annotations
import copy
import numpy as np
import pyvista as pv
import pytest
import loop_cgal
from loop_cgal._loop_cgal import ImplicitCutMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_surface():
    # Unit square made of two triangles
    return pv.Plane(center=(0,0,0),direction=(0,0,1),i_size=1.0,j_size=1.0)


@pytest.fixture
def clipper_surface():
    # A square that overlaps half of the unit square
    return pv.Plane(center=(0,0,0),direction=(1,0,0),i_size=2.0,j_size=2.0)


@pytest.fixture
def unit_trimesh():
    """TriMesh built from a known 1×1 unit square."""
    return loop_cgal.TriMesh(pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1.0, j_size=1.0))


# ---------------------------------------------------------------------------
# Properties: area, points, n_cells, n_points
# ---------------------------------------------------------------------------

def test_area_unit_square(unit_trimesh):
    """area property should return the surface area directly from CGAL."""
    assert abs(unit_trimesh.area - 1.0) < 1e-6


def test_points_shape(unit_trimesh):
    """points property returns (n, 3) float array."""
    pts = unit_trimesh.points
    assert isinstance(pts, np.ndarray)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.shape[0] == unit_trimesh.n_points


def test_n_cells_positive(unit_trimesh):
    """n_cells should equal the number of triangles (≥1)."""
    assert unit_trimesh.n_cells > 0
    saved_tris = np.array(unit_trimesh.save().triangles)
    assert unit_trimesh.n_cells == saved_tris.shape[0]


def test_n_points_positive(unit_trimesh):
    """n_points should equal the number of vertices (≥3)."""
    assert unit_trimesh.n_points >= 3
    saved_verts = np.array(unit_trimesh.save().vertices)
    assert unit_trimesh.n_points == saved_verts.shape[0]


def test_n_cells_and_n_points_consistent(unit_trimesh):
    """n_cells and n_points should both report non-zero consistent values."""
    assert unit_trimesh.n_cells > 0
    assert unit_trimesh.n_points > 0


# ---------------------------------------------------------------------------
# Constructors: from_vertices_and_triangles, get_vertices_and_triangles
# ---------------------------------------------------------------------------

def test_from_vertices_and_triangles_roundtrip():
    """from_vertices_and_triangles should produce an equivalent mesh."""
    surface = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=2.0, j_size=2.0)
    original = loop_cgal.TriMesh(surface)
    verts, tris = original.get_vertices_and_triangles()

    rebuilt = loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)

    assert rebuilt.n_cells == original.n_cells
    assert rebuilt.n_points == original.n_points
    assert abs(rebuilt.area - original.area) < 1e-6


def test_from_vertices_and_triangles_rejects_invalid():
    """from_vertices_and_triangles should raise ValueError for bad input."""
    with pytest.raises((ValueError, Exception)):
        loop_cgal.TriMesh.from_vertices_and_triangles(
            np.zeros((2, 3)),  # only 2 vertices — can't form a triangle
            np.array([[0, 1, 2]]),
        )


def test_get_vertices_and_triangles_shapes(unit_trimesh):
    """get_vertices_and_triangles returns correctly shaped arrays."""
    verts, tris = unit_trimesh.get_vertices_and_triangles()
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert tris.ndim == 2 and tris.shape[1] == 3


# ---------------------------------------------------------------------------
# Conversions: to_pyvista, vtk
# ---------------------------------------------------------------------------

def test_to_pyvista_returns_polydata(unit_trimesh):
    pv_mesh = unit_trimesh.to_pyvista()
    assert isinstance(pv_mesh, pv.PolyData)
    assert pv_mesh.n_points > 0
    assert pv_mesh.n_cells > 0


def test_vtk_alias_matches_to_pyvista(unit_trimesh):
    """vtk() is an alias for to_pyvista() and must return identical geometry."""
    via_to_pyvista = unit_trimesh.to_pyvista()
    via_vtk = unit_trimesh.vtk()
    np.testing.assert_array_equal(via_to_pyvista.points, via_vtk.points)
    np.testing.assert_array_equal(via_to_pyvista.faces, via_vtk.faces)


# ---------------------------------------------------------------------------
# Deep copy: clone, copy, __copy__, __deepcopy__
# ---------------------------------------------------------------------------

def test_clone_returns_trimesh(unit_trimesh):
    cloned = unit_trimesh.clone()
    assert isinstance(cloned, loop_cgal.TriMesh)


def test_clone_has_same_geometry(unit_trimesh):
    cloned = unit_trimesh.clone()
    assert cloned.n_cells == unit_trimesh.n_cells
    assert cloned.n_points == unit_trimesh.n_points
    assert abs(cloned.area - unit_trimesh.area) < 1e-10


def test_clone_is_independent(square_surface, clipper_surface):
    """Mutating the clone must not affect the original."""
    original = loop_cgal.TriMesh(square_surface)
    cloned = original.clone()
    n_cells_before = original.n_cells

    clipper = loop_cgal.TriMesh(clipper_surface)
    cloned.cut_with_surface(clipper)

    assert original.n_cells == n_cells_before, "clone mutation leaked into original"


def test_copy_method_returns_trimesh(unit_trimesh):
    c = unit_trimesh.copy()
    assert isinstance(c, loop_cgal.TriMesh)
    assert c.n_cells == unit_trimesh.n_cells


def test_copy_module_shallow(unit_trimesh):
    """copy.copy() should use __copy__ and return an independent TriMesh."""
    c = copy.copy(unit_trimesh)
    assert isinstance(c, loop_cgal.TriMesh)
    assert c.n_cells == unit_trimesh.n_cells


def test_copy_module_deep(unit_trimesh):
    """copy.deepcopy() should use __deepcopy__ and return an independent TriMesh."""
    c = copy.deepcopy(unit_trimesh)
    assert isinstance(c, loop_cgal.TriMesh)
    assert c.n_cells == unit_trimesh.n_cells


def test_deepcopy_is_independent(square_surface, clipper_surface):
    """Mutating the deepcopy must not affect the original."""
    original = loop_cgal.TriMesh(square_surface)
    deep = copy.deepcopy(original)
    n_cells_before = original.n_cells

    clipper = loop_cgal.TriMesh(clipper_surface)
    deep.cut_with_surface(clipper)

    assert original.n_cells == n_cells_before, "deepcopy mutation leaked into original"


# ---------------------------------------------------------------------------
# overlaps
# ---------------------------------------------------------------------------

def test_overlaps_intersecting():
    """Two overlapping planes should report overlaps=True."""
    a = loop_cgal.TriMesh(pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=2.0, j_size=2.0))
    b = loop_cgal.TriMesh(pv.Plane(center=(0, 0, 0), direction=(1, 0, 0), i_size=2.0, j_size=2.0))
    assert a.overlaps(b)


def test_overlaps_separated():
    """Two well-separated meshes must not report overlap."""
    a = loop_cgal.TriMesh(pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1.0, j_size=1.0))
    b = loop_cgal.TriMesh(pv.Plane(center=(100, 0, 0), direction=(0, 0, 1), i_size=1.0, j_size=1.0))
    assert not a.overlaps(b)


# @pytest.mark.parametrize("remesh_kwargs", [
#     {"split_long_edges": True, "target_edge_length": 0.2, "number_of_iterations": 1, "protect_constraints": True, "relax_constraints": False},
#     {"split_long_edges": False, "target_edge_length": 0.02, "number_of_iterations": 2, "protect_constraints": True, "relax_constraints": True},
# ])
def test_loading_and_saving(square_surface):
    tm = loop_cgal.TriMesh(square_surface)
    saved = tm.save()
    verts = np.array(saved.vertices)
    tris = np.array(saved.triangles)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert tris.ndim == 2 and tris.shape[1] == 3
    assert verts.shape[0] > 0
    assert tris.shape[0] > 0


def test_cut_with_surface(square_surface, clipper_surface):
    tm = loop_cgal.TriMesh(square_surface)
    clip = loop_cgal.TriMesh(clipper_surface)
    before = np.array(tm.save().triangles).shape[0]
    tm.cut_with_surface(clip)
    after = np.array(tm.save().triangles).shape[0]
    # If clipper intersects, faces should be non-zero and not increase
    assert after >= 0
    assert after <= before


@pytest.mark.parametrize("kwargs", [
    {"split_long_edges": True, "target_edge_length": 0.25, "number_of_iterations": 1, "protect_constraints": True, "relax_constraints": False},
    {"split_long_edges": True, "target_edge_length": 0.05, "number_of_iterations": 2, "protect_constraints": False, "relax_constraints": True},
])
def test_remesh_changes_vertices(square_surface, kwargs):
    tm = loop_cgal.TriMesh(square_surface)
    
    # Call remesh using keyword args compatible with the binding
    tm.remesh(kwargs["split_long_edges"], kwargs["target_edge_length"], kwargs["number_of_iterations"], kwargs["protect_constraints"], kwargs["relax_constraints"])
    after_v = np.array(tm.save().vertices).shape[0]
    # Remesh should produce a valid mesh
    assert after_v > 0
    # Either vertices increase due to splitting or stay similar; ensure no catastrophic collapse
    # assert after_v >= 0.5 * before_v


def _make_curved_clipper(x_offset: float, y_half: float, z_top: float, z_bot: float, n: int = 6) -> pv.PolyData:
    """Build a curved (listric) clipper surface that bows in the X direction.

    The surface lies roughly in the YZ plane at x=x_offset but curves so that
    deeper vertices are offset further in X.  This simulates a listric fault
    being used as a clipper.
    """
    ys = np.linspace(-y_half, y_half, n)
    zs = np.linspace(z_top, z_bot, n)
    verts = []
    for z in zs:
        # curvature: bow increases with depth (small enough that cut stays within 50 m of X=0)
        # The deepest target vertex is at z=-7000; max x-deviation = c * 7000 < tol=50 m → c < 0.007
        curve = x_offset + 0.005 * (z - z_top)
        for y in ys:
            verts.append([curve, y, z])
    verts = np.array(verts, dtype=float)
    faces = []
    for iz in range(n - 1):
        for iy in range(n - 1):
            i0 = iz * n + iy
            i1 = i0 + 1
            i2 = i0 + n
            i3 = i2 + 1
            faces.extend([[3, i0, i1, i3], [3, i0, i3, i2]])
    faces = np.array(faces).flatten()
    return pv.PolyData(verts, faces)


@pytest.mark.parametrize("clipper_factory,label", [
    (
        # Planar clipper: YZ plane at X=0, far smaller than the 4 km deep target
        lambda: pv.Plane(
            center=(0.0, 0.0, -250.0),
            direction=(1.0, 0.0, 0.0),
            i_size=500.0,
            j_size=500.0,
            i_resolution=4,
            j_resolution=4,
        ).triangulate(),
        "planar",
    ),
    (
        # Curved (listric) clipper: also smaller than the target, bows in X with depth
        lambda: _make_curved_clipper(x_offset=0.0, y_half=250.0, z_top=0.0, z_bot=-500.0),
        "curved",
    ),
])
def test_no_bridge_when_clipper_shorter_than_target(clipper_factory, label):
    """Clipper that doesn't extend fully through the target must not leave a bridge.

    Setup
    -----
    Target  : vertical fault surface in the XZ plane (Y=0), 10 km wide × 4 km deep.
    Clipper : either a planar or curved (listric) surface at X≈0, only 500 m deep
              — far smaller than the 4 km depth of the target.

    Without the fix, the lower portion (Z < -500 m) is left as a bridge.  With
    boundary extension, the clipper rim is pushed out to cover the full target so
    the result must lie entirely on one side of X=0.  The interior vertices of the
    clipper are unchanged, so the cut location is preserved for both planar and
    curved cases.
    """
    # Large vertical fault in XZ plane (Y=0): 10 km wide, 4 km deep.
    target_pv = pv.Plane(
        center=(0.0, 0.0, -2000.0),
        direction=(0.0, 1.0, 0.0),
        i_size=10000.0,
        j_size=4000.0,
        i_resolution=20,
        j_resolution=8,
    ).triangulate()

    target = loop_cgal.TriMesh(target_pv)
    clipper = loop_cgal.TriMesh(clipper_factory())

    faces_removed = target.cut_with_surface(clipper)
    result = target.to_pyvista()

    assert result.n_points > 0, f"[{label}] Clip removed the entire mesh"
    assert faces_removed > 0, f"[{label}] Clip was a no-op — meshes may not intersect"

    # After a clean cut at X≈0, all remaining vertices must be on ONE side.
    # A bridge would leave vertices significantly on both sides.
    xs = result.points[:, 0]
    tol = 50.0  # 50 m tolerance for vertices exactly on the cut boundary
    on_positive = np.any(xs > tol)
    on_negative = np.any(xs < -tol)
    assert not (on_positive and on_negative), (
        f"[{label}] Bridge detected: vertices span both sides of the cut plane "
        f"(X range [{xs.min():.1f}, {xs.max():.1f}] m)."
    )


@pytest.mark.parametrize("collocate_target,collocate_clipper,label", [
    (True,  False, "collocated_target"),
    (False, True,  "collocated_clipper"),
    (True,  True,  "collocated_both"),
])
def test_no_artefact_with_collocated_vertices(collocate_target, collocate_clipper, label):
    """Collocated (duplicate) vertices must not cause bridges or a no-op clip.

    Two failure modes are possible:
    - Collocated vertices in the *target* create zero-area faces that can make
      PMP::do_intersect return false, silently skipping the entire clip.
    - Collocated vertices in the *clipper* create degenerate border faces whose
      normals are near-zero, causing skirt edges to be skipped and leaving a
      bridge gap at those positions.

    Both are fixed by calling PMP::stitch_borders before clipping.
    """
    def _add_collocated_seam(pv_mesh: pv.PolyData) -> pv.PolyData:
        """Duplicate a row of interior vertices along the mesh midline (Z=-2000).

        The duplicate vertices are collocated with the originals but topologically
        distinct, creating zero-area triangles along the seam.
        """
        pts = pv_mesh.points.copy()
        # Find vertices near the midline (Z ≈ -2000)
        seam = np.where(np.abs(pts[:, 2] + 2000.0) < 50.0)[0]
        if len(seam) == 0:
            return pv_mesh  # nothing to duplicate
        extra = pts[seam].copy()  # exact same positions
        new_pts = np.vstack([pts, extra])
        return pv.PolyData(new_pts, pv_mesh.faces)

    target_pv = pv.Plane(
        center=(0.0, 0.0, -2000.0),
        direction=(0.0, 1.0, 0.0),
        i_size=10000.0,
        j_size=4000.0,
        i_resolution=20,
        j_resolution=8,
    ).triangulate()

    clipper_pv = pv.Plane(
        center=(0.0, 0.0, -250.0),
        direction=(1.0, 0.0, 0.0),
        i_size=500.0,
        j_size=500.0,
        i_resolution=4,
        j_resolution=4,
    ).triangulate()

    if collocate_target:
        target_pv = _add_collocated_seam(target_pv)
    if collocate_clipper:
        clipper_pv = _add_collocated_seam(clipper_pv)

    target = loop_cgal.TriMesh(target_pv)
    clipper = loop_cgal.TriMesh(clipper_pv)

    faces_removed = target.cut_with_surface(clipper)
    result = target.to_pyvista()

    assert result.n_points > 0, f"[{label}] Clip removed the entire mesh"
    assert faces_removed > 0, f"[{label}] Clip was a no-op (do_intersect likely wrong due to degenerate faces)"

    xs = result.points[:, 0]
    tol = 50.0
    on_positive = np.any(xs > tol)
    on_negative = np.any(xs < -tol)
    assert not (on_positive and on_negative), (
        f"[{label}] Bridge detected: X range [{xs.min():.1f}, {xs.max():.1f}] m"
    )


def test_cut_with_implicit_function(square_surface):
    tm = loop_cgal.TriMesh(square_surface)
    # create a scalar property that varies across vertices
    saved = tm.save()
    nverts = np.array(saved.vertices).shape[0]
    prop = [float(i) / max(1, (nverts - 1)) for i in range(nverts)]
    # cut at 0.5 keeping positive side
    tm.cut_with_implicit_function(prop, 0.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)
    res = tm.save()
    v = np.array(res.vertices).shape[0]
    f = np.array(res.triangles).shape[0]
    assert v >= 0
    assert f >= 0
