#pragma once

#include <vector>
#include <unordered_map>
#include <array>
#include <unordered_map>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

namespace std
{
    template <>
    struct hash<std::array<int, 6>>
    {
        std::size_t operator()(const std::array<int, 6> &arr) const
        {
            std::size_t seed = 0;
            for (int val : arr)
            {
                seed ^= std::hash<int>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}
struct GridCell
{
    int i, j, k; // Cell indices

    bool operator==(const GridCell &other) const
    {
        return i == other.i && j == other.j && k == other.k;
    }
};

// Hash function for GridCell
struct GridCellHash
{
    std::size_t operator()(const GridCell &cell) const
    {
        return std::hash<int>()(cell.i) ^
               std::hash<int>()(cell.j) ^
               std::hash<int>()(cell.k);
    }
};
typedef std::unordered_set<GridCell, GridCellHash> active_cells_set;
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;

typedef std::vector<Point> Triangle; // Assuming Triangle is a vector of points
std::vector<Triangle> marchingCubes(const std::array<double,8> &values, double isovalue,
                                      double origin_x, double origin_y, double origin_z,
                                      double step_x, double step_y, double step_z
                                      );

int compute_cube_index(const std::array<double, 8> &cube_values, double isovalue);
Point interpolate_vertex(const Point &p1, const Point &p2, double val1, double val2, double isovalue);
