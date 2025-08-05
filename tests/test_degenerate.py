import numpy as np
import loop_cgal
import pyvista as pv
def test_degenerate_triangles():
    """Test handling of degenerate triangles in TriMesh."""
    # Create a surface with degenerate triangles
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 4], [4, 0, 0]])  # Degenerate triangle (4 is not a valid vertex)
    
    surface = pv.PolyData.from_regular_faces(points, faces)
    
    try:
        tri_mesh = loop_cgal.TriMesh(surface)
        print("TriMesh created successfully with degenerate triangles.")
    except ValueError as e:
        print(f"ValueError: {e}")


if __name__ == "__main__":
    loop_cgal.set_verbose(True)
    test_degenerate_triangles()
    print("Test completed.")