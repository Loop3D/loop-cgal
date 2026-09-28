"""Regression tests for Loop3D/loop-cgal#14.

Two coupled defects, both exercised here:

1. ``clip_with_plane`` was not idempotent. Re-clipping a mesh that already lay
   entirely inside the halfspace removed no face but still corefined against the
   plane, splitting every edge along the existing seam. Repeating a plane grew
   the mesh without bound -- the "long runtime" half of the issue.
2. The exact round trip rebuilt the mesh with no deduplication, so the surplus
   seam points rounded onto coincident doubles. ``PMP::clip`` cannot cope with
   those, which is the reported segfault (observed on Linux/x86-64 only).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

import loop_cgal


def raw_mesh(mesh):
    """Vertices/triangles with the export dedup disabled, so this reports the
    mesh's own vertex set rather than a cleaned copy of it."""
    return mesh.get_vertices_and_triangles(area_threshold=0.0, duplicate_vertex_threshold=0.0)


def n_coincident(vertices):
    return len(vertices) - len(np.unique(vertices, axis=0))


@pytest.fixture
def bumpy_surface():
    """Irregular, non-axis-aligned surface: clip intersections land on
    high-bit-depth rationals, which is what makes the rounding collide."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, size=(250, 2))
    z = 1.5 * np.sin(pts[:, 0] * 0.7) + 1.1 * np.cos(pts[:, 1] * 0.5) + rng.normal(0, 0.05, 250)
    return loop_cgal.TriMesh.from_vertices_and_triangles(
        np.column_stack([pts, z]).astype(np.float64),
        Delaunay(pts).simplices.astype(np.int32),
    )


# The plane that cuts the fixture in two.
PLANE = (-0.5063653104002493, -0.45214174227658877, -0.7342765264628509, -0.3550784848736548)


def test_first_clip_removes_faces(bumpy_surface):
    """Guard must not suppress a clip that has something to remove."""
    before = bumpy_surface.n_cells
    assert bumpy_surface.clip_with_plane(*PLANE) > 0
    assert bumpy_surface.n_cells < before


@pytest.mark.parametrize("use_exact_kernel", [True, False])
def test_repeated_clip_is_idempotent(bumpy_surface, use_exact_kernel):
    """The same plane twice is a no-op: 0 faces removed, mesh untouched."""
    bumpy_surface.clip_with_plane(*PLANE, use_exact_kernel)
    settled_v, settled_t = raw_mesh(bumpy_surface)

    for _ in range(20):
        assert bumpy_surface.clip_with_plane(*PLANE, use_exact_kernel) == 0

    v, t = raw_mesh(bumpy_surface)
    np.testing.assert_array_equal(v, settled_v)
    np.testing.assert_array_equal(t, settled_t)


@pytest.mark.parametrize("use_exact_kernel", [True, False])
def test_repeated_clip_breeds_no_coincident_vertices(bumpy_surface, use_exact_kernel):
    for _ in range(20):
        bumpy_surface.clip_with_plane(*PLANE, use_exact_kernel)
        v, _ = raw_mesh(bumpy_surface)
        assert n_coincident(v) == 0


def test_cycled_planes_reach_a_fixed_point(bumpy_surface):
    """The faultgen shape: a pool of near-tangent planes applied over and over.
    Mesh size must settle rather than grow with every pass."""
    rng = np.random.default_rng(3)
    v0, _ = raw_mesh(bumpy_surface)
    planes = []
    for _ in range(8):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        offset = -float(np.quantile(v0 @ normal, 0.97))
        planes.append(tuple(float(x) for x in normal) + (offset,))

    for i in range(len(planes) * 2):
        bumpy_surface.clip_with_plane(*planes[i % len(planes)])
    settled = bumpy_surface.n_points

    for i in range(len(planes) * 3):
        bumpy_surface.clip_with_plane(*planes[i % len(planes)])
        assert n_coincident(raw_mesh(bumpy_surface)[0]) == 0
    assert bumpy_surface.n_points == settled


def test_coincident_input_vertices_are_merged():
    """A mesh that arrives already carrying duplicates -- read_from_file bypasses
    export_mesh's dedup -- must be cleaned on the way into the exact kernel."""
    verts = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]],
        dtype=np.float64,
    )  # rows 4 and 5 duplicate rows 1 and 2
    tris = np.array([[0, 1, 2], [4, 3, 5]], dtype=np.int32)
    mesh = loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)
    assert n_coincident(raw_mesh(mesh)[0]) == 2

    mesh.clip_with_plane(1.0, 0.0, 0.0, -0.75)  # keeps x <= 0.75
    assert n_coincident(raw_mesh(mesh)[0]) == 0


def test_degenerate_plane_normal_raises(bumpy_surface):
    with pytest.raises(ValueError):
        bumpy_surface.clip_with_plane(0.0, 0.0, 0.0, 1.0)
