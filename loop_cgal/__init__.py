from .loop_cgal import clip_surface, NumpyMesh, NumpyPlane, clip_plane, corefine_mesh
from .loop_cgal import TriMesh as _TriMesh
from .loop_cgal import Grid as _Grid
import pyvista as pv
import numpy as np
from typing import Tuple

class TriMesh(_TriMesh):
    """
    A class for handling triangular meshes using CGAL.
    
    Inherits from the base TriMesh class and provides additional functionality.
    """
    def __init__(self, surface: pv.PolyData = None, vertices: np.ndarray = None, triangles: np.ndarray = None):
        if surface is not None:
            if not isinstance(surface, pv.PolyData):
                raise TypeError("surface must be a pyvista.PolyData object")
            
            vertices = np.array(surface.points).copy()
            triangles = surface.faces.reshape(-1, 4)[:, 1:].copy()
            
        super().__init__(vertices, triangles)
    @classmethod
    def from_cpp(cls, mesh):
        """
        Create a TriMesh instance from a C++ mesh object.
        
        Parameters
        ----------
        mesh : C++ mesh object
            The mesh object to convert.
        
        Returns
        -------
        TriMesh
            The converted TriMesh object.
        """
        
        numpymesh = mesh.save(1e-10,1e-10,False)

        return cls(None,numpymesh.vertices, numpymesh.triangles)
    def to_pyvista(self, area_threshold: float = 1e-6,  # this is the area threshold for the faces, if the area is smaller than this it will be removed
            duplicate_vertex_threshold: float = 1e-4,  # this is the threshold for duplicate vertices
            verbose: bool = False) -> pv.PolyData:
        """
        Convert the TriMesh to a pyvista PolyData object.
        
        Returns
        -------
        pyvista.PolyData
            The converted PolyData object.
        """
        np_mesh = self.save(area_threshold, duplicate_vertex_threshold, verbose)
        vertices = np.array(np_mesh.vertices).copy()
        triangles = np.array(np_mesh.triangles).copy()
        return pv.PolyData.from_regular_faces(vertices, triangles)
class Grid(_Grid):
    """
    A class for creating and manipulating a grid of points in 3D space.
    
    Inherits from the base Grid class and provides additional functionality.
    """
    def __init__(self, origin_x: float, origin_y: float, origin_z: float,
                 step_x: float, step_y: float, step_z: float,
                 nsteps_x: int, nsteps_y: int, nsteps_z: int):
        super().__init__(origin_x, origin_y, origin_z, step_x, step_y, step_z,
                         nsteps_x, nsteps_y, nsteps_z)
    def extract_isosurface(self, name: str, isovalue: float) -> TriMesh:
        """
        Extract an isosurface from the grid based on a scalar field.
        
        Parameters
        ----------
        name : str
            The name of the scalar field to use for isosurface extraction.
        isovalue : float
            The isovalue for the isosurface extraction.
        
        Returns
        -------
        TriMesh
            The extracted isosurface as a TriMesh object.
        """
        np_mesh = self._extract_isosurface(name, isovalue)
        vertices = np.array(np_mesh.vertices).copy()
        triangles = np.array(np_mesh.triangles).copy()
        return pv.PolyData.from_regular_faces(vertices, triangles)
        
def clip_pyvista_polydata_with_plane(
    surface: pv.PolyData,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    target_edge_length: float = 10.0,
    remesh_before_clipping: bool = True,
    remesh_after_clipping: bool = True,
    remove_degenerate_faces: bool = True,
    duplicate_vertex_threshold: float = 0.001,
    area_threshold: float = 0.0001,
    protect_constraints: bool = False,
    relax_constraints: bool = True,
    verbose: bool = False,
) -> pv.PolyData:
    """
    Clip a pyvista PolyData object with a plane using the CGAL library.
    Parameters
    ----------
    surface : pyvista.PolyData

        The surface to be clipped.
    plane_origin : np.ndarray
        The origin point of the clipping plane.
    plane_normal : np.ndarray
        The normal vector of the clipping plane.
    target_edge_length : float, optional
        The target edge length for the remeshing process, by default 10.0
    remesh_before_clipping : bool, optional
        Whether to remesh the surface before clipping, by default True
    remesh_after_clipping : bool, optional
        Whether to remesh the surface after clipping, by default True
    remove_degenerate_faces : bool, optional
        Whether to remove degenerate faces from the resulting mesh, by default True
    duplicate_vertex_threshold : float, optional
        The threshold for merging duplicate vertices, by default 0.001
    area_threshold : float, optional
        The area threshold for removing small faces, by default 0.0001
    verbose : bool, optional
        Whether to print verbose output, by default False
    Returns
    -------
    pyvista.PolyData
        The resulting clipped surface.
    """
    surface = surface.triangulate()
    tm = NumpyMesh()
    tm.vertices = np.array(surface.points).copy()
    tm.triangles = surface.faces.reshape(-1, 4)[:, 1:].copy()
    plane = NumpyPlane()
    plane.origin = np.asarray(plane_origin, dtype=np.float64)
    plane.normal = np.asarray(plane_normal, dtype=np.float64)

    mesh = clip_plane(
        tm,
        plane,
        target_edge_length=target_edge_length,
        remesh_before_clipping=remesh_before_clipping,
        remesh_after_clipping=remesh_after_clipping,
        remove_degenerate_faces=remove_degenerate_faces,
        duplicate_vertex_threshold=duplicate_vertex_threshold,
        area_threshold=area_threshold,
        protect_constraints=protect_constraints,
        relax_constraints=relax_constraints,
        verbose=verbose,
    )
    return pv.PolyData.from_regular_faces(mesh.vertices, mesh.triangles)


def clip_pyvista_polydata(
    surface_1: pv.PolyData,
    surface_2: pv.PolyData,
    target_edge_length: float = 10.0,
    remesh_before_clipping: bool = True,
    remesh_after_clipping: bool = True,
    remove_degenerate_faces: bool = True,
    duplicate_vertex_threshold: float = 0.001,
    area_threshold: float = 0.0001,
    protect_constraints: bool = False,
    relax_constraints: bool = True,
    verbose: bool = False,
) -> pv.PolyData:
    """
    Clip two pyvista PolyData objects using the CGAL library.

    Parameters
    ----------
    surface_1 : pyvista.PolyData
        The first surface to be clipped.
    surface_2 : pyvista.PolyData
        The second surface to be used for clipping.

    Returns
    -------
    pyvista.PolyData
        The resulting clipped surface.
    """
    surface_1 = surface_1.triangulate()
    surface_2 = surface_2.triangulate()
    tm = NumpyMesh()
    tm.vertices = np.array(surface_1.points).copy()
    tm.triangles = surface_1.faces.reshape(-1, 4)[:, 1:].copy()
    clipper = NumpyMesh()
    clipper.vertices = np.array(surface_2.points).copy()
    clipper.triangles = surface_2.faces.reshape(-1, 4)[:, 1:].copy()
    mesh = clip_surface(
        tm,
        clipper,
        target_edge_length=target_edge_length,
        remesh_before_clipping=remesh_before_clipping,
        remesh_after_clipping=remesh_after_clipping,
        remove_degenerate_faces=remove_degenerate_faces,
        duplicate_vertex_threshold=duplicate_vertex_threshold,
        area_threshold=area_threshold,
        protect_constraints=protect_constraints,
        relax_constraints=relax_constraints,
        verbose=verbose,
    )
    out = pv.PolyData.from_regular_faces(mesh.vertices, mesh.triangles)

    return out


def corefine_pyvista_polydata(
    surface_1: pv.PolyData,
    surface_2: pv.PolyData,
    target_edge_length: float = 10.0,
    duplicate_vertex_threshold: float = 0.001,
    area_threshold: float = 0.0001,
    number_of_iterations: int = 10,
    protect_constraints: bool = True,
    relax_constraints: bool = True,
    verbose: bool = False,
) -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Corefine two pyvista PolyData objects using the CGAL library.

    Parameters
    ----------
    surface_1 : pyvista.PolyData
        The first surface to be cored.
    surface_2 : pyvista.PolyData
        The second surface to be used for cording.

    Returns
    -------
    pyvista.PolyData
        The resulting cored surface.
    """
    surface_1 = surface_1.triangulate()
    surface_2 = surface_2.triangulate()
    tm1 = NumpyMesh()
    tm1.vertices = np.array(surface_1.points).copy()
    tm1.triangles = surface_1.faces.reshape(-1, 4)[:, 1:].copy()
    tm2 = NumpyMesh()
    tm2.vertices = np.array(surface_2.points).copy()
    tm2.triangles = surface_2.faces.reshape(-1, 4)[:, 1:].copy()

    tm1, tm2 = corefine_mesh(
        tm1,
        tm2,
        target_edge_length=target_edge_length,
        duplicate_vertex_threshold=duplicate_vertex_threshold,
        area_threshold=area_threshold,
        number_of_iterations=number_of_iterations,
        relax_constraints=relax_constraints,
        protect_constraints=protect_constraints,
        verbose=verbose,
    )
    return (
        pv.PolyData.from_regular_faces(tm1.vertices, tm1.triangles),
        pv.PolyData.from_regular_faces(tm2.vertices, tm2.triangles),
    )
