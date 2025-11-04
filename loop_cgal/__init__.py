from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse as sp
import pyvista as pv

from ._loop_cgal import TriMesh as _TriMesh
from ._loop_cgal import verbose  # noqa: F401
from ._loop_cgal import set_verbose as set_verbose

from .utils import validate_pyvista_polydata, validate_vertices_and_faces


class TriMesh(_TriMesh):
    """
    A class for handling triangular meshes using CGAL.

    Inherits from the base TriMesh class and provides additional functionality.
    """

    def __init__(self, surface: pv.PolyData):
        # Validate input surface
        validate_pyvista_polydata(surface, "input surface")

        # Triangulate to ensure we have triangular faces
        surface = surface.triangulate()

        # Extract vertices and triangles
        verts = np.array(surface.points, dtype=np.float64).copy()
        faces = surface.faces.reshape(-1, 4)[:, 1:].copy().astype(np.int32)
        if not validate_vertices_and_faces(verts, faces):
            raise ValueError("Invalid surface geometry")

        super().__init__(verts, faces)
    @classmethod
    def from_vertices_and_triangles(
        cls, vertices: np.ndarray, triangles: np.ndarray
    ) -> TriMesh:
        """
        Create a TriMesh from vertices and triangle indices.

        Parameters
        ----------
        vertices : np.ndarray
            An array of shape (n_vertices, 3) containing the vertex coordinates.
        triangles : np.ndarray
            An array of shape (n_triangles, 3) containing the triangle vertex indices.

        Returns
        -------
        TriMesh
            The created TriMesh object.
        """
        # Create a temporary PyVista PolyData object for validation
        if not validate_vertices_and_faces(vertices, triangles):
            raise ValueError("Invalid vertices or triangles")
        surface = pv.PolyData(vertices, np.hstack((np.full((triangles.shape[0], 1), 3), triangles)).flatten())
        return cls(surface)

    def get_vertices_and_triangles(
        self,
        area_threshold: float = 1e-6,  # this is the area threshold for the faces, if the area is smaller than this it will be removed
        duplicate_vertex_threshold: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the vertices and triangle indices of the TriMesh.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - An array of shape (n_vertices, 3) with the vertex coordinates.
            - An array of shape (n_triangles, 3) with the triangle vertex indices.
        """
        np_mesh = self.save(area_threshold, duplicate_vertex_threshold)
        vertices = np.array(np_mesh.vertices).copy()
        triangles = np.array(np_mesh.triangles).copy()
        return vertices, triangles

    def to_pyvista(
        self,
        area_threshold: float = 1e-6,  # this is the area threshold for the faces, if the area is smaller than this it will be removed
        duplicate_vertex_threshold: float = 1e-4,  # this is the threshold for duplicate vertices
    ) -> pv.PolyData:
        """
        Convert the TriMesh to a pyvista PolyData object.

        Returns
        -------
        pyvista.PolyData
            The converted PolyData object.
        """
        np_mesh = self.save(area_threshold, duplicate_vertex_threshold)
        vertices = np.array(np_mesh.vertices).copy()
        triangles = np.array(np_mesh.triangles).copy()
        return pv.PolyData.from_regular_faces(vertices, triangles)

    def vtk(
        self,
        area_threshold: float = 1e-6,
        duplicate_vertex_threshold: float = 1e-4,
    ) -> pv.PolyData:
        """
        Alias for to_pyvista method.
        """
        return self.to_pyvista(area_threshold, duplicate_vertex_threshold)

    def copy(self) -> TriMesh:
        """
        Create a copy of the TriMesh.

        Returns
        -------
        TriMesh
            A copy of the TriMesh object.
        """
        return TriMesh(self.to_pyvista())
