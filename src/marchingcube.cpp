#include "marchingcubes.h"
#include "edge_table.h"

int compute_cube_index(const std::array<double, 8> &cube_values, double isovalue)
{
    // Compute cube index based on scalar values
    int index = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (cube_values[i] < isovalue)
        {
            index |= (1 << i);
        }
    }
    return index;
}

Point interpolate_vertex(const Point &p1, const Point &p2, double val1, double val2, double isovalue)
{
    // Linear interpolation of vertex position
    double t = (isovalue - val1) / (val2 - val1);
    return Point(p1.x() + t * (p2.x() - p1.x()),
                 p1.y() + t * (p2.y() - p1.y()),
                 p1.z() + t * (p2.z() - p1.z()));
}
std::vector<Triangle> marchingCubes(const std::array<double, 8> &values, double isovalue,
                                    double origin_x, double origin_y, double origin_z,
                                    double step_x, double step_y, double step_z)
{
    std::vector<Triangle> triangle_soup;
    // Initialize the grid origin and spacing
    Point grid_origin(origin_x, origin_y, origin_z);
    Vector grid_spacing(step_x, step_y, step_z);
    std::array<Point, 8> cube_corners = {
        grid_origin,
        grid_origin + Vector(step_x, 0, 0),
        grid_origin + Vector(step_x, step_y, 0),
        grid_origin + Vector(0, step_y, 0),
        grid_origin + Vector(0, 0, step_z),
        grid_origin + Vector(step_x, 0, step_z),
        grid_origin + Vector(step_x, step_y, step_z),
        grid_origin + Vector(0, step_y, step_z)};

    int cube_index = compute_cube_index(values, isovalue);
    // If the cube is entirely inside or outside the isosurface, skip it
    if (cube_index == 0 || cube_index == 255)
    {
        return triangle_soup;
    }
    else
    {
        const int edges = edgeTable[cube_index]; // edgeTable is a predefined lookup table

        // Interpolate the vertices on the intersected edges
        std::array<Point, 12> edge_vertices;
        for (int i = 0; i < 12; ++i)
        {
            if (edges & (1 << i))
            {
                int v1 = edgeConnection[i][0]; // edgeConnection maps edges to corner indices
                int v2 = edgeConnection[i][1];
                edge_vertices[i] = interpolate_vertex(cube_corners[v1], cube_corners[v2], values[v1], values[v2], isovalue);
            }
        }

        // Generate triangles for the current cube
        const auto &triangles = triTable[cube_index]; // triTable is a predefined lookup table
        Triangle triangle;

        for (int i = 0; triangles[i] != -1; i += 3)
        {

            triangle.push_back(edge_vertices[triangles[i]]);
            triangle.push_back(edge_vertices[triangles[i + 1]]);
            triangle.push_back(edge_vertices[triangles[i + 2]]);
            triangle_soup.push_back(triangle);
            triangle.clear();
        }
    }
    return triangle_soup;
}
