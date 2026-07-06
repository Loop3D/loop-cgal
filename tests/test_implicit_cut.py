"""Behavioural tests for ``cut_with_implicit_function``.

These replace an earlier smoke test whose only assertions were ``n_verts >= 0``
and ``n_faces >= 0`` — conditions that are true for *any* mesh and so could
never fail. The tests below pin the actual geometry of the isocontour cut:

* the seam is inserted exactly on the isovalue,
* ``KEEP_POSITIVE_SIDE`` / ``KEEP_NEGATIVE_SIDE`` retain only their side
  (no fringe of wrong-side triangles clinging to the seam),
* ``PRESERVE_INTERSECTION`` keeps the whole surface but still inserts the seam,
* NaN (off-extent) property values leave their triangles untouched,
* ``snap_tol`` reuses an existing vertex instead of inserting a near-duplicate.

The property is chosen to equal the vertex x-coordinate, so the isocontour
``property == value`` is the plane ``x == value`` and every geometric claim can
be checked directly against vertex coordinates.
"""
from __future__ import annotations

import numpy as np
import pytest

import loop_cgal
from loop_cgal._loop_cgal import ImplicitCutMode

SIZE = 10.0
EPS = 1e-6


def _grid(nx: int = 11, ny: int = 11, size: float = SIZE):
    """A flat (z=0) triangulated grid spanning [0, size] in x and y."""
    verts = np.array(
        [[size * i / (nx - 1), size * j / (ny - 1), 0.0]
         for j in range(ny) for i in range(nx)],
        dtype=np.float64,
    )

    def vid(i, j):
        return j * nx + i

    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            tris += [
                [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)],
                [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)],
            ]
    return verts, np.array(tris, dtype=np.int32)


def _cut(prop, value, mode, snap_tol=1e-4):
    """Run the cut on a fresh grid whose property equals its x-coordinate."""
    verts, tris = _grid()
    tm = loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)
    if prop == "x":
        prop = verts[:, 0].tolist()
    tm.cut_with_implicit_function(prop, value, mode, snap_tol)
    v, f = tm.get_vertices_and_triangles()
    return tm, v, f


def _used_x(v, f):
    """x-coordinates of only the vertices actually referenced by a face.

    ``get_vertices_and_triangles`` also returns vertices left isolated by the
    cut, so filtering to referenced vertices is what tells us where the surface
    actually reaches.
    """
    return v[np.unique(f), 0]


def test_seam_inserted_exactly_on_isovalue():
    """A cut between grid lines inserts seam vertices lying exactly on the plane x==value."""
    _, v, f = _cut("x", 4.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)
    xs = _used_x(v, f)
    assert np.any(np.isclose(xs, 4.5, atol=EPS)), "no seam vertices were inserted on the isovalue"


def test_no_degenerate_faces():
    """The cut must never emit a face with a repeated vertex index."""
    _, _, f = _cut("x", 4.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)
    assert len(f) > 0
    for tri in f:
        assert tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2], (
            f"degenerate face emitted: {tri}"
        )


def test_preserve_intersection_keeps_full_area_and_inserts_seam():
    """PRESERVE_INTERSECTION splits along the seam but keeps the whole surface."""
    tm, v, f = _cut("x", 4.5, ImplicitCutMode.PRESERVE_INTERSECTION)
    assert tm.area == pytest.approx(SIZE * SIZE, rel=1e-6), "PRESERVE_INTERSECTION must keep the full area"
    xs = _used_x(v, f)
    assert np.any(np.isclose(xs, 4.5, atol=EPS)), "seam must still be inserted"
    assert xs.min() < 4.5 < xs.max(), "both sides must be retained"


def test_nan_triangles_are_kept_untouched():
    """Triangles with a NaN (off-extent) property survive regardless of side/value.

    The far-negative region (x < 3, marked NaN) must remain even under
    KEEP_POSITIVE_SIDE with a value well above it.
    """
    verts, tris = _grid()
    prop = verts[:, 0].copy()
    prop[verts[:, 0] < 3.0] = np.nan
    tm = loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)
    tm.cut_with_implicit_function(prop.tolist(), 4.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)
    v, f = tm.get_vertices_and_triangles()
    xs = _used_x(v, f)
    assert xs.min() < EPS, "NaN-property region (x≈0) was dropped but should be kept untouched"


def test_snap_tol_reuses_nearby_vertex():
    """A value within snap_tol of a grid line inserts no near-duplicate vertices."""
    verts, tris = _grid()
    tm = loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)
    n_before = tm.n_points
    # 4.0 is a grid line; 4.0 + 1e-6 is far inside snap_tol (1e-4 of a unit edge).
    tm.cut_with_implicit_function(
        verts[:, 0].tolist(), 4.0 + 1e-6, ImplicitCutMode.KEEP_POSITIVE_SIDE, 1e-4
    )
    assert tm.n_points == n_before, (
        f"snap_tol should reuse existing vertices, but point count grew "
        f"{n_before} -> {tm.n_points}"
    )


def test_keep_positive_side_is_tight():
    """KEEP_POSITIVE_SIDE must retain exactly the region x >= value."""
    tm, v, f = _cut("x", 4.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)
    xs = _used_x(v, f)
    assert xs.min() >= 4.5 - EPS, (
        f"kept a wrong-side fringe: min referenced x = {xs.min():.4f} (< 4.5)"
    )
    assert tm.area == pytest.approx((SIZE - 4.5) * SIZE, rel=1e-3)


def test_keep_negative_side_is_tight():
    """KEEP_NEGATIVE_SIDE must retain exactly the region x <= value."""
    tm, v, f = _cut("x", 4.5, ImplicitCutMode.KEEP_NEGATIVE_SIDE)
    xs = _used_x(v, f)
    assert xs.max() <= 4.5 + EPS, (
        f"kept a wrong-side fringe: max referenced x = {xs.max():.4f} (> 4.5)"
    )
    assert tm.area == pytest.approx(4.5 * SIZE, rel=1e-3)


def test_sides_partition_the_original():
    """Positive and negative sides must partition the original area (they only
    share the zero-area seam), so their areas sum to the whole."""
    pos, _, _ = _cut("x", 4.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)
    neg, _, _ = _cut("x", 4.5, ImplicitCutMode.KEEP_NEGATIVE_SIDE)
    assert pos.area + neg.area == pytest.approx(SIZE * SIZE, rel=1e-3)
