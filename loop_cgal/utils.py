import pyvista as pv
import numpy as np
import scipy.sparse as sp
def validate_pyvista_polydata(
    surface: pv.PolyData, surface_name: str = "surface"
) -> None:
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


def validate_vertices_and_faces(verts, faces):
    """Validate vertices and faces arrays.
    Parameters
    ----------
    verts : np.ndarray

        An array of shape (n_vertices, 3) containing the vertex coordinates.
    faces : np.ndarray

        An array of shape (n_faces, 3) containing the triangle vertex indices.
    Raises
    ------
    ValueError
        If the vertices or faces are invalid.
    """
    if type(verts) is not np.ndarray:
        try:
            verts = np.array(verts)
        except Exception:
            raise ValueError("Vertices must be a numpy array")
    if type(faces) is not np.ndarray:
        try:
            faces = np.array(faces)
        except Exception:
            raise ValueError("Faces must be a numpy array")
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
        raise ValueError(
            f"Surface triangle indices exceed vertex count (max index: {faces.max()}, vertex count: {verts.shape[0]})"
        )
    # Check for degenerate triangles
    # build a ntris x nverts matrix
    # populate with true for vertex in each triangle
    # sum rows and if not equal to 3 then it is degenerate
    face_idx = np.arange(faces.shape[0])
    face_idx = np.tile(face_idx, (3, 1)).T.flatten()
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
        raise ValueError(
            f"Surface contains degenerate triangles: {degen_idx} (each triangle must have exactly 3 vertices)"
        )
    return True