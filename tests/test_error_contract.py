"""Error-handling contract for the mutating operations.

The cut/clip/corefine operations now *raise* on a genuine failure (invalid or
empty input, or an operation that was expected to succeed but did not) instead
of printing to stderr and returning 0. A return of 0 is reserved for the one
legitimate no-op: a cut whose meshes do not intersect.

pybind11 maps C++ std::invalid_argument -> ValueError and std::runtime_error /
other std::exception -> RuntimeError.
"""
from __future__ import annotations

import numpy as np
import pytest

import loop_cgal
from loop_cgal._loop_cgal import ImplicitCutMode
from loop_cgal._loop_cgal import TriMesh as RawTriMesh


def _unit_square():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return loop_cgal.TriMesh.from_vertices_and_triangles(verts, tris)


def _empty_mesh():
    """A TriMesh with no faces (built directly via the C++ ctor, bypassing
    the Python validation that would otherwise reject it)."""
    return RawTriMesh(np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32))


def test_cut_with_surface_empty_clipper_raises():
    tm = _unit_square()
    with pytest.raises(ValueError, match="empty"):
        tm.cut_with_surface(_empty_mesh())


def test_cut_with_surface_empty_source_raises():
    empty = _empty_mesh()
    with pytest.raises(ValueError, match="empty"):
        empty.cut_with_surface(_unit_square())


def test_clip_with_plane_empty_mesh_raises():
    empty = _empty_mesh()
    with pytest.raises(ValueError, match="empty"):
        empty.clip_with_plane(1.0, 0.0, 0.0, 0.0)


def test_cut_with_implicit_function_property_size_mismatch_raises():
    tm = _unit_square()
    wrong = [0.0, 1.0]  # mesh has 4 vertices
    with pytest.raises(ValueError, match="does not match"):
        tm.cut_with_implicit_function(wrong, 0.5, ImplicitCutMode.KEEP_POSITIVE_SIDE)


def test_non_intersecting_cut_is_a_no_op_not_an_error():
    """Disjoint meshes are the one legitimate no-op: return 0, do not raise."""
    tm = _unit_square()
    far = np.array(
        [[10, 10, 0], [11, 10, 0], [11, 11, 0], [10, 11, 0]], dtype=np.float64
    )
    clipper = loop_cgal.TriMesh.from_vertices_and_triangles(
        far, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    )
    removed = tm.cut_with_surface(clipper)
    assert removed == 0
    assert tm.n_cells == 2  # mesh untouched
