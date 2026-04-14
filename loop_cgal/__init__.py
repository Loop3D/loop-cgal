from __future__ import annotations

import copy
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

    def __init__(self, surface):
        if isinstance(surface, _TriMesh):
            # Copy-construct directly from another TriMesh at the C++ level.
            # This is used by read_from_file and anywhere a TriMesh-to-TriMesh
            # copy is needed without a pyvista round-trip.
            super().__init__(surface)
            return

        # Validate input surface
        validate_pyvista_polydata(surface, "input surface")

        # Triangulate to ensure we have triangular faces
        if not surface.is_all_triangles:
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
        surface = pv.PolyData(
            vertices,
            np.hstack((np.full((triangles.shape[0], 1), 3), triangles)).flatten(),
        )
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

    @property
    def area(self) -> float:
        """Surface area computed directly from the CGAL mesh (no pyvista conversion)."""
        return self._cgal_area()

    @property
    def points(self) -> np.ndarray:
        """Vertex coordinates as (n, 3) numpy array (no pyvista conversion)."""
        return self._cgal_points()

    @property
    def n_cells(self) -> int:
        """Number of faces in the CGAL mesh (no pyvista conversion)."""
        return self._cgal_n_faces()

    @property
    def n_points(self) -> int:
        """Number of vertices in the CGAL mesh (no pyvista conversion)."""
        return self._cgal_n_vertices()

    @classmethod
    def read_from_file(cls, path: str) -> "TriMesh":
        """Read a mesh from a binary file written by :meth:`write_to_file`.

        Bypasses the pyvista/VTK stack entirely — much faster for temp files
        passed between CGAL operations.
        """
        return cls(_TriMesh._read_from_file(path))

    def clone(self) -> TriMesh:
        """
        Return a deep copy of this TriMesh as a Python ``TriMesh`` instance.

        Uses the C++ copy-constructor binding to clone the CGAL mesh directly,
        with no numpy array roundtrip.
        """
        return TriMesh(self)

    def copy(self, deep: bool = True) -> TriMesh:
        """
        Return a deep copy of this TriMesh.

        Uses the C++-level ``clone()`` to copy the CGAL mesh directly without
        a pyvista round-trip, preserving all vertices, faces, and fixed edges.

        Returns
        -------
        TriMesh
            A deep copy of this TriMesh.
        """
        return self.clone()

    def __copy__(self) -> TriMesh:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> TriMesh:
        result = self.clone()
        memo[id(self)] = result
        return result
