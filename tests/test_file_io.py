"""Tests for TriMesh native binary file I/O (write_to_file / read_from_file).

These tests isolate the LCMESH round-trip and the clone/copy paths that
depend on the pybind11 copy-constructor, both of which were implicated in
a segfault caused by calling _TriMesh.__init__ on an uninitialized holder.
"""
from __future__ import annotations

import copy

import numpy as np
import pyvista as pv
import pytest

import loop_cgal
from loop_cgal import TriMesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_plane():
    """1×1 unit square as pv.PolyData."""
    return pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1.0, j_size=1.0)


@pytest.fixture
def unit_mesh(simple_plane):
    return TriMesh(simple_plane)


@pytest.fixture
def large_plane():
    """Denser mesh for round-trip fidelity checks."""
    return pv.Plane(
        center=(0, 0, -500),
        direction=(0, 1, 0),
        i_size=2000.0,
        j_size=1000.0,
        i_resolution=10,
        j_resolution=5,
    ).triangulate()


# ---------------------------------------------------------------------------
# write_to_file / read_from_file round-trip
# ---------------------------------------------------------------------------


def test_write_creates_file(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    import os
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_roundtrip_n_points(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert loaded.n_points == unit_mesh.n_points


def test_roundtrip_n_cells(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert loaded.n_cells == unit_mesh.n_cells


def test_roundtrip_area(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert abs(loaded.area - unit_mesh.area) < 1e-9


def test_roundtrip_vertices(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    orig_pts = np.sort(unit_mesh.points, axis=0)
    load_pts = np.sort(loaded.points, axis=0)
    np.testing.assert_allclose(orig_pts, load_pts, atol=1e-10)


def test_roundtrip_returns_trimesh_subclass(tmp_path, unit_mesh):
    """read_from_file must return a Python TriMesh, not a bare _TriMesh."""
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert type(loaded) is TriMesh


def test_roundtrip_large_mesh(tmp_path, large_plane):
    mesh = TriMesh(large_plane)
    path = str(tmp_path / "large.lcm")
    mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert loaded.n_points == mesh.n_points
    assert loaded.n_cells == mesh.n_cells
    assert abs(loaded.area - mesh.area) < 1e-4


def test_read_bad_magic(tmp_path):
    bad = tmp_path / "bad.lcm"
    bad.write_bytes(b"BADMAG\x00\x00\x00\x00\x00\x00\x00\x00")
    with pytest.raises(Exception, match="[Bb]ad magic|cannot open|LCMESH"):
        TriMesh.read_from_file(str(bad))


def test_read_missing_file(tmp_path):
    with pytest.raises(Exception):
        TriMesh.read_from_file(str(tmp_path / "nonexistent.lcm"))


# ---------------------------------------------------------------------------
# Loaded mesh is fully usable (no dangling state)
# ---------------------------------------------------------------------------


def test_loaded_mesh_to_pyvista(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    pv_mesh = loaded.to_pyvista()
    assert isinstance(pv_mesh, pv.PolyData)
    assert pv_mesh.n_points > 0


def test_loaded_mesh_remesh(tmp_path, large_plane):
    """read_from_file result can be remeshed without crashing."""
    mesh = TriMesh(large_plane)
    path = str(tmp_path / "mesh.lcm")
    mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    loaded.remesh(target_edge_length=200.0, number_of_iterations=1)
    assert loaded.n_cells > 0


def test_loaded_mesh_cut_with_surface(tmp_path, large_plane):
    """read_from_file result can be cut without crashing."""
    mesh = TriMesh(large_plane)
    path = str(tmp_path / "mesh.lcm")
    mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)

    clipper_pv = pv.Plane(
        center=(0, 0, -500),
        direction=(1, 0, 0),
        i_size=3000.0,
        j_size=1500.0,
        i_resolution=4,
        j_resolution=4,
    ).triangulate()
    clipper = TriMesh(clipper_pv)
    loaded.cut_with_surface(clipper)
    assert loaded.n_cells >= 0  # just must not crash


# ---------------------------------------------------------------------------
# clone / copy after file round-trip
# ---------------------------------------------------------------------------


def test_clone_of_loaded_mesh(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    cloned = loaded.clone()
    assert type(cloned) is TriMesh
    assert cloned.n_cells == loaded.n_cells
    assert cloned.n_points == loaded.n_points


def test_clone_of_loaded_is_independent(tmp_path, large_plane):
    mesh = TriMesh(large_plane)
    path = str(tmp_path / "mesh.lcm")
    mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    cloned = loaded.clone()
    n_before = loaded.n_cells

    clipper_pv = pv.Plane(
        center=(0, 0, -500),
        direction=(1, 0, 0),
        i_size=3000.0,
        j_size=1500.0,
    ).triangulate()
    clipper = TriMesh(clipper_pv)
    cloned.cut_with_surface(clipper)

    assert loaded.n_cells == n_before, "clone mutation leaked into loaded mesh"


def test_deepcopy_of_loaded_mesh(tmp_path, unit_mesh):
    path = str(tmp_path / "mesh.lcm")
    unit_mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    deep = copy.deepcopy(loaded)
    assert type(deep) is TriMesh
    assert deep.n_cells == loaded.n_cells


def test_write_then_read_then_write(tmp_path, unit_mesh):
    """Double round-trip must be stable."""
    p1 = str(tmp_path / "a.lcm")
    p2 = str(tmp_path / "b.lcm")
    unit_mesh.write_to_file(p1)
    loaded = TriMesh.read_from_file(p1)
    loaded.write_to_file(p2)
    reloaded = TriMesh.read_from_file(p2)
    assert reloaded.n_points == unit_mesh.n_points
    assert reloaded.n_cells == unit_mesh.n_cells
    assert abs(reloaded.area - unit_mesh.area) < 1e-9


# ---------------------------------------------------------------------------
# write_to_file after mesh mutation (tombstoned vertices/faces)
# ---------------------------------------------------------------------------


def test_write_after_remesh(tmp_path, large_plane):
    """write_to_file must handle tombstoned CGAL entries after remesh."""
    mesh = TriMesh(large_plane)
    mesh.remesh(target_edge_length=150.0, number_of_iterations=1)
    path = str(tmp_path / "remeshed.lcm")
    mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert loaded.n_cells == mesh.n_cells
    assert loaded.n_points == mesh.n_points
    assert abs(loaded.area - mesh.area) < 1e-3


def test_write_after_cut(tmp_path, large_plane):
    """write_to_file must handle tombstoned entries after cut_with_surface."""
    mesh = TriMesh(large_plane)
    clipper_pv = pv.Plane(
        center=(0, 0, -500),
        direction=(1, 0, 0),
        i_size=3000.0,
        j_size=1500.0,
    ).triangulate()
    clipper = TriMesh(clipper_pv)
    mesh.cut_with_surface(clipper)

    path = str(tmp_path / "cut.lcm")
    mesh.write_to_file(path)
    loaded = TriMesh.read_from_file(path)
    assert loaded.n_cells == mesh.n_cells
    assert loaded.n_points == mesh.n_points
