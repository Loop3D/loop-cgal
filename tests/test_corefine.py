"""Tests for ``TriMesh.corefine``.

Corefinement inserts the shared intersection polyline into *both* meshes so that
patches taken from each side (e.g. a contact cut by a fault and the matching
fault patch) stitch together watertight. The contract we pin here is exactly
that: after corefining two surfaces that cross along a line, both meshes carry
the *same* set of vertices along that line.

Two crossing grids are used: ``A`` lies in the z=0 plane, ``B`` in the y=0
plane, so their intersection is the segment y=0, z=0 parametrised by x.

Note: only the default exact-kernel path is exercised. ``use_exact_kernel=False``
runs CGAL corefinement on the inexact kernel, which hard-crashes (SIGBUS) on
these open surfaces — see ``test_corefine_inexact_kernel_is_unsafe`` below. A
native crash cannot be trapped by ``xfail`` (it takes down the whole pytest
process), so that path is documented and skipped rather than executed.
"""
from __future__ import annotations

import numpy as np
import pytest

import loop_cgal

EXT = 1.0
N = 6


def _grid_plane(fixed_axis: str):
    """A triangulated grid in [-EXT, EXT]^2, embedded in the plane fixed_axis==0."""
    a = np.linspace(-EXT, EXT, N)
    b = np.linspace(-EXT, EXT, N)
    if fixed_axis == "z":
        verts = np.array([[x, y, 0.0] for y in b for x in a], dtype=np.float64)
    elif fixed_axis == "y":
        verts = np.array([[x, 0.0, z] for z in b for x in a], dtype=np.float64)
    else:
        raise ValueError(fixed_axis)

    def vid(i, j):
        return j * N + i

    tris = []
    for j in range(N - 1):
        for i in range(N - 1):
            tris += [
                [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)],
                [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)],
            ]
    return verts, np.array(tris, dtype=np.int32)


def _seam_x(points):
    """x-coordinates of vertices lying on the intersection line y=0, z=0."""
    on = (np.abs(points[:, 1]) < 1e-9) & (np.abs(points[:, 2]) < 1e-9)
    return set(np.round(points[on, 0], 9))


def _crossing_pair():
    A = loop_cgal.TriMesh.from_vertices_and_triangles(*_grid_plane("z"))
    B = loop_cgal.TriMesh.from_vertices_and_triangles(*_grid_plane("y"))
    return A, B


def test_corefine_adds_vertices_and_returns_the_count():
    """corefine returns the number of vertices it added to this mesh."""
    A, B = _crossing_pair()
    before = A.n_points
    added = A.corefine(B)
    assert added == A.n_points - before
    assert added > 0, "crossing surfaces must gain seam vertices"


def test_corefine_inserts_coincident_polyline_into_both_meshes():
    """The shared intersection polyline must be identical on both meshes."""
    A, B = _crossing_pair()
    A.corefine(B)
    pa, _ = A.get_vertices_and_triangles()
    pb, _ = B.get_vertices_and_triangles()
    seam_a = _seam_x(pa)
    seam_b = _seam_x(pb)
    assert len(seam_a) > 2, "expected an inserted polyline, not just the original corners"
    assert seam_a == seam_b, (
        f"seam vertices differ between meshes — not watertight:\n"
        f"  A: {sorted(seam_a)}\n  B: {sorted(seam_b)}"
    )


def test_corefine_mutates_both_meshes():
    """Both meshes are corefined in place (the second argument is not read-only)."""
    A, B = _crossing_pair()
    b_before = B.n_points
    A.corefine(B)
    assert B.n_points > b_before, "the other mesh must also receive the shared polyline"


def test_corefine_preserves_area():
    """Corefinement only subdivides — it must not change either surface's area."""
    A, B = _crossing_pair()
    area_a, area_b = A.area, B.area
    A.corefine(B)
    assert A.area == pytest.approx(area_a, rel=1e-6)
    assert B.area == pytest.approx(area_b, rel=1e-6)


@pytest.mark.skip(
    reason="corefine(use_exact_kernel=False) hard-crashes (SIGBUS) on open "
    "intersecting surfaces; a native crash would take down the pytest process. "
    "Tracked separately — the inexact path needs input validation or removal."
)
def test_corefine_inexact_kernel_is_unsafe():
    A, B = _crossing_pair()
    A.corefine(B, use_exact_kernel=False)
