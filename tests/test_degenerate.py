import numpy as np
import pyvista as pv
import pytest
import loop_cgal


def make_invalid_polydata_invalid_index():
    # face references a non-existent vertex index (4)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    return pv.PolyData.from_regular_faces(points, faces)


def make_polydata_degenerate_triangle():
    # triangle repeats a vertex index -> degenerate
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 1]])
    return pv.PolyData.from_regular_faces(points, faces)


def test_invalid_index_raises_value_error():
    surface = make_invalid_polydata_invalid_index()
    with pytest.raises(ValueError, match="exceed vertex count"):
        _ = loop_cgal.TriMesh(surface)


def test_degenerate_triangle_raises_value_error():
    surface = make_polydata_degenerate_triangle()
    with pytest.raises(ValueError, match="degenerate triangles"):
        _ = loop_cgal.TriMesh(surface)