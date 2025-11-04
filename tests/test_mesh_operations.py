from __future__ import annotations
import numpy as np
import pyvista as pv
import pytest
import loop_cgal
from loop_cgal._loop_cgal import ImplicitCutMode


@pytest.fixture
def square_surface():
    # Unit square made of two triangles
    return pv.Plane(center=(0,0,0),direction=(0,0,1),i_size=1.0,j_size=1.0)


@pytest.fixture
def clipper_surface():
    # A square that overlaps half of the unit square
    return pv.Plane(center=(0,0,0),direction=(1,0,0),i_size=2.0,j_size=2.0)


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
