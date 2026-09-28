"""Regression test for the implicit-cut snap dropping kept surface.

The snap recorded "this vertex is the crossing" by writing the isovalue into the
shared per-vertex array. That is a per-edge fact stored per-vertex, so every
sub-triangle incident to the vertex then read it as lying on the seam, and an
inclusive keep/discard test dropped legitimate keep-side sub-triangles from BOTH
sides -- silently deleting surface wherever the seam ran through a vertex.
"""

from __future__ import annotations

import numpy as np
import pytest

import loop_cgal
from loop_cgal._loop_cgal import ImplicitCutMode


# --- 1. snapping must not delete kept surface ------------------------------

# Two triangles sharing edge 0-2. Vertex 0 sits a hair off the isosurface, and
# its neighbour 3 is far away, so the crossing on edge 0-3 lands within snap_tol
# of vertex 0 and snaps onto it.
SNAP_VERTS = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
SNAP_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
SNAP_PROP = [1e-3, 1e-3, -1e-3, -100.0]


def _cut_area(mode, snap_tol):
    m = loop_cgal.TriMesh.from_vertices_and_triangles(SNAP_VERTS, SNAP_TRIS)
    m.cut_with_implicit_function(SNAP_PROP, 0.0, mode, snap_tol)
    return m.area


@pytest.mark.parametrize(
    "mode", [ImplicitCutMode.KEEP_POSITIVE_SIDE, ImplicitCutMode.KEEP_NEGATIVE_SIDE]
)
def test_snapping_does_not_change_the_kept_area(mode):
    """Snapping is a meshing detail; it must not move the cut."""
    assert _cut_area(mode, 1e-4) == pytest.approx(_cut_area(mode, 0.0), abs=1e-5)


def test_the_two_sides_partition_the_surface():
    """Every sub-triangle belongs to exactly one side — none dropped, none double
    counted, even where the seam runs through an existing vertex."""
    total = 1.0  # the two input triangles
    kept = _cut_area(ImplicitCutMode.KEEP_POSITIVE_SIDE, 1e-4) + _cut_area(
        ImplicitCutMode.KEEP_NEGATIVE_SIDE, 1e-4
    )
    assert kept == pytest.approx(total, abs=1e-5)
