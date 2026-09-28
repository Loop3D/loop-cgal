"""Regression tests: user-added edge constraints must survive a cut and be
honoured by a subsequent protected remesh.

Background
----------
``add_fixed_edges`` records interior (non-border) edges that must be preserved
when the mesh is remeshed. ``cut_with_surface`` / ``clip_with_plane`` mutate the
mesh via CGAL's clip, which tombstones/garbage-collects elements and (for the
exact-kernel path) rebuilds the mesh from scratch — both of which invalidate the
``Edge_index`` handles stored for the constraints. Before the fix the constraint
set was never refreshed, so after a cut it either silently pointed at recycled
edges or was thrown away entirely, and the subsequent ``remesh`` freely erased
the constrained polyline.

Discriminator
-------------
A constrained edge may legitimately be *split* into collinear sub-edges by
remesh (CGAL keeps the sub-edges constrained), but the polyline itself must stay
exactly on its original line with both endpoints intact. When the constraint is
lost the region is remeshed freely and no mesh edges remain on the line. We
therefore count mesh edges lying exactly on the constrained segment: preserved →
a connected chain covering the full span; lost → zero.
"""
from __future__ import annotations
import numpy as np
import pytest
import pyvista as pv
import loop_cgal


NX, NY = 11, 4  # grid columns x=0..10, rows y=0..3 (rows 1,2 are interior)
# Fully interior vertical edge at x=8 spanning y=1..2 (both endpoints interior).
EDGE_X = 8.0
EDGE_Y0, EDGE_Y1 = 1.0, 2.0
A_COORD = np.array([EDGE_X, EDGE_Y0, 0.0])
B_COORD = np.array([EDGE_X, EDGE_Y1, 0.0])


def _build_grid():
    verts = np.array(
        [[float(i), float(j), 0.0] for j in range(NY) for i in range(NX)],
        dtype=np.float64,
    )

    def vid(i, j):
        return j * NX + i

    tris = []
    for j in range(NY - 1):
        for i in range(NX - 1):
            a, b = vid(i, j), vid(i + 1, j)
            c, d = vid(i + 1, j + 1), vid(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    return verts, np.array(tris, dtype=np.int32), vid


def _on_line(p):
    """True if point p lies exactly on the constrained segment (x=8, z=0, 1<=y<=2)."""
    return (
        abs(p[0] - EDGE_X) < 1e-9
        and abs(p[2]) < 1e-9
        and (EDGE_Y0 - 1e-9) <= p[1] <= (EDGE_Y1 + 1e-9)
    )


def _line_edges(verts, tris):
    """Return the set of mesh edges (as index pairs) lying on the constrained segment."""
    edges = set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if _on_line(verts[a]) and _on_line(verts[b]):
                edges.add((min(int(a), int(b)), max(int(a), int(b))))
    return edges


def _has_coord(verts, coord):
    return bool(np.any(np.all(np.isclose(verts, coord), axis=1)))


def _cut_and_remesh(add_constraint, method, use_exact_kernel):
    verts, tris, vid = _build_grid()
    tm = loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)
    if add_constraint:
        tm.add_fixed_edges(np.array([[vid(8, 1), vid(8, 2)]], dtype=np.int32))

    if method == "surface":
        # Vertical clipper at x=5, normal -x, so the side x>5 (containing the
        # x=8 constrained edge) is kept. The cut boundary is near the edge but
        # does not cross it.
        clipper = pv.Plane(
            center=(5.0, 1.5, 0.0), direction=(-1.0, 0.0, 0.0),
            i_size=8.0, j_size=8.0, i_resolution=4, j_resolution=4,
        ).triangulate()
        removed = tm.cut_with_surface(
            loop_cgal.TriMesh(clipper), use_exact_kernel=use_exact_kernel
        )
        assert removed > 0, "cut removed nothing — clipper did not intersect"
    else:  # plane: keep -x+5 < 0  <=>  x > 5
        removed = tm.clip_with_plane(-1.0, 0.0, 0.0, 5.0, use_exact_kernel=use_exact_kernel)
        assert removed > 0, "plane clip removed nothing"

    # target_edge_length (0.3) is shorter than the 1.0-long constrained edge, so
    # a lost constraint would let remesh freely re-triangulate the region.
    tm.remesh(True, 0.3, 3, True, False)
    return tm.get_vertices_and_triangles()


@pytest.mark.parametrize("use_exact_kernel", [True, False])
@pytest.mark.parametrize("method", ["surface", "plane"])
def test_user_constraint_survives_cut_and_remesh(method, use_exact_kernel):
    """A user-added interior constraint must survive the cut and be honoured by remesh."""
    verts, tris = _cut_and_remesh(True, method, use_exact_kernel)

    # 1. Both original endpoints are still present, exactly (constrained edge
    #    endpoints are never moved by smoothing).
    assert _has_coord(verts, A_COORD), f"[{method}/{use_exact_kernel}] endpoint (8,1,0) lost"
    assert _has_coord(verts, B_COORD), f"[{method}/{use_exact_kernel}] endpoint (8,2,0) lost"

    # 2. The constrained polyline is tiled by mesh edges lying exactly on the
    #    line — a lost constraint leaves zero.
    edges = _line_edges(verts, tris)
    assert len(edges) >= 1, (
        f"[{method}/{use_exact_kernel}] no mesh edges on the constrained line — "
        f"constraint was not preserved through the cut"
    )

    # 3. The chain covers the full span y=1..2 (endpoints connected through
    #    collinear sub-vertices, all exactly on the line).
    on_line_ys = sorted(
        p[1] for p in verts if _on_line(p)
    )
    assert on_line_ys[0] == pytest.approx(EDGE_Y0)
    assert on_line_ys[-1] == pytest.approx(EDGE_Y1)


@pytest.mark.parametrize("use_exact_kernel", [True, False])
@pytest.mark.parametrize("method", ["surface", "plane"])
def test_constraint_makes_a_difference(method, use_exact_kernel):
    """Control: the same cut+remesh without the constraint erases the polyline.

    This pins the discriminating power of the test above — the only difference
    between the two runs is the ``add_fixed_edges`` call.
    """
    _, tris_with = _cut_and_remesh(True, method, use_exact_kernel)
    verts_with, tris_with = _cut_and_remesh(True, method, use_exact_kernel)
    verts_without, tris_without = _cut_and_remesh(False, method, use_exact_kernel)

    with_edges = len(_line_edges(verts_with, tris_with))
    without_edges = len(_line_edges(verts_without, tris_without))
    assert with_edges > without_edges, (
        f"[{method}/{use_exact_kernel}] constraint made no difference "
        f"(with={with_edges}, without={without_edges}) — it is not being honoured"
    )
