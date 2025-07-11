from loop_cgal import Grid

grid = Grid(0, 0, 0, 1, 1, 1, 10, 10, 10)
grid.add_scalar_field("temperature", [i for i in range(10) for j in range(10) for k in range(10)])
mesh = grid.extract_isosurface( "temperature", 1.0)
mesh.plot(show_edges=True, show_scalar_bar=True)