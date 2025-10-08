from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse as sp
import pyvista as pv

from ._loop_cgal import TriMesh as _TriMesh
from ._loop_cgal import verbose # noqa: F401
from ._loop_cgal import set_verbose as set_verbose





def validate_pyvista_polydata(surface: pv.PolyData, surface_name: str = "surface") -> None:
    """Validate a PyVista PolyData object.
    
    Parameters
    ----------
    surface : pv.PolyData
        The surface to validate
    surface_name : str
        Name of the surface for error messages
        
    Raises
    ------
    ValueError
        If the surface is invalid
    """
    if not isinstance(surface, pv.PolyData):
        raise ValueError(f"{surface_name} must be a pyvista.PolyData object")
    
    if surface.n_points == 0:
        raise ValueError(f"{surface_name} has no points")
    
    if surface.n_cells == 0:
        raise ValueError(f"{surface_name} has no cells")
    
    points = np.asarray(surface.points)
    if not np.isfinite(points).all():
        raise ValueError(f"{surface_name} points contain NaN or infinite values")



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
        
        # Additional validation on extracted data
        if verts.size == 0:
            raise ValueError("Surface has no vertices after triangulation")
        
        if faces.size == 0:
            raise ValueError("Surface has no triangular faces after triangulation")
        
        if not np.isfinite(verts).all():
            raise ValueError("Surface vertices contain NaN or infinite values")
        
        # Check triangle indices
        max_vertex_index = verts.shape[0] - 1
        if faces.min() < 0:
            raise ValueError("Surface has negative triangle indices")
        
        if faces.max() > max_vertex_index:
            raise ValueError(f"Surface triangle indices exceed vertex count (max index: {faces.max()}, vertex count: {verts.shape[0]})")
        # Check for degenerate triangles
        # build a ntris x nverts matrix 
        # populate with true for vertex in each triangle
        # sum rows and if not equal to 3 then it is degenerate
        face_idx = np.arange(faces.shape[0])
        face_idx = np.tile(face_idx, (3,1)).T.flatten()
        faces_flat = faces.flatten()
        m = sp.coo_matrix(
            (np.ones(faces_flat.shape[0]), (faces_flat, face_idx)),
            shape=(verts.shape[0], faces.shape[0]),
            dtype=bool,
        )
        # coo duplicates entries so just make sure its boolean
        m = m > 0
        if not np.all(m.sum(axis=0) == 3):
            degen_idx = np.where(m.sum(axis=0) != 3)[1]
            raise ValueError(f"Surface contains degenerate triangles: {degen_idx} (each triangle must have exactly 3 vertices)")
        

        super().__init__(verts, faces)
        
    def to_pyvista(self, area_threshold: float = 1e-6,  # this is the area threshold for the faces, if the area is smaller than this it will be removed
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

