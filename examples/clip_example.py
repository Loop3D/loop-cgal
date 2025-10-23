import loop_cgal
import numpy as np
import pyvista as pv
from LoopStructural.datatypes import BoundingBox
def test():
    bb = BoundingBox(np.zeros(3), np.ones(3))
    grid = bb.structured_grid().vtk()
    print(grid)
    grid["scalars"] = grid.points[:,0]
    surface = grid.contour([.5])
    surface_1_tri = surface.faces.reshape(-1, 4)[:, 1:].copy()
    surface_1_verts = surface.points.copy()
    grid["scalars"] = grid.points[:,1]
    surface_2 = grid.contour([.5])
    surface_2_tri = surface_2.faces.reshape(-1, 4)[:, 1:].copy()
    surface_2_verts = surface_2.points.copy()

    surface_clipped = loop_cgal.clip_pyvista_polydata(
        surface, surface_2,
        target_edge_length=0.2,
        remesh_before_clipping=True,
        remesh_after_clipping=True,
        remove_degenerate_faces=False,
        duplicate_vertex_threshold=0.000001,
        area_threshold=0.0001,
        protect_constraints=True,
        relax_constraints=False,
        verbose=True
    )  # surface_1_tri, surface_1_verts, surface_2_tri, surface_2_verts
    # )
    print(surface_clipped)
    # print(mesh.vertices.shape)
    # print(surface_1_verts.shape)
    # print(mesh.triangles)
    # print(mesh.triangles, mesh.vertices)
    # surface_clipped = pv.PolyData.from_regular_faces(mesh.vertices, mesh.triangles)
    p = pv.Plotter()
    # p.add_mesh(surface)
    p.add_mesh(surface_2, color="blue", show_edges=True)
    p.add_mesh(surface_clipped, color="red", show_edges=True)
    # p.add_points(mesh.vertices, render_points_as_spheres=True, point_size=5)
    p.show()
if __name__ == "__main__":
    test()
