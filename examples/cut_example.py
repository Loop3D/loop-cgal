import numpy as np
from loop_cgal import TriMesh, set_verbose
from LoopStructural.datatypes import BoundingBox
set_verbose(True)


bb = BoundingBox(np.zeros(3), np.ones(3))
grid = bb.structured_grid().vtk()
grid["scalars"] = grid.points[:, 0]
surface = grid.contour([0.5])
property = surface.points[:, 1].copy()
surface['prop'] = property
surface.save("input_mesh.vtk")

mesh = TriMesh(surface)

# Define a scalar property that is the Y coordinate of each vertex

# Cut value at y = 0.6; this will split triangles crossing y=0.6
cut_value = property.mean()
# mesh.remesh(0.10, protect_constraints=False)

mesh.vtk().save('before_cut.vtk')
# Invoke the new method (property is python list or numpy array)
mesh.cut_with_implicit_function(property.tolist(), cut_value)
mesh.reverse_face_orientation()
# Convert back to pyvista for visualization
out = mesh.to_pyvista()
out.save('cut_mesh.vtk')
# # Visualize the result
# pl = pv.Plotter()
# pl.add_mesh(out, show_edges=True, color='lightgray')
# pl.add_mesh(pv.PolyData(np.array([[cut_value, -1e3, 0.0],[cut_value,1e3,0.0]])), color='red')
# pl.show()
