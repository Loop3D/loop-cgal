#include "grid.h"
#include <CGAL/Polygon_mesh_processing/autorefinement.h>

namespace PMP = CGAL::Polygon_mesh_processing;

Grid::Grid(double origin_x, double origin_y, double origin_z, 
           double step_x, double step_y, double step_z, 
           int nsteps_x, int nsteps_y, int nsteps_z)
    : _origin_x(origin_x), _origin_y(origin_y), _origin_z(origin_z),
      _step_x(step_x), _step_y(step_y), _step_z(step_z),
      _nsteps_x(nsteps_x), _nsteps_y(nsteps_y), _nsteps_z(nsteps_z) {}

void Grid::addScalarField(const std::string &name, const std::vector<double> &values) {
    if (values.size() != _nsteps_x * _nsteps_y * _nsteps_z) {
        throw std::invalid_argument("Values size does not match grid dimensions.");
    }
    _scalar_fields[name] = values;
}
std::array<double, 8> Grid::get_cell_values(int i, int j, int k, std::string name) const{
// Extract the scalar field values for the specified cell
    std::array<double,8> cell_values;

    std::array<int,8> xcorner = {0, 1, 0, 1, 0, 1, 0, 1};
    std::array<int,8> ycorner = {0, 0, 1, 1, 0, 0, 1, 1};
    std::array<int,8> zcorner = {0, 0, 0, 0, 1, 1, 1, 1};
    for (int ci = 0; ci < 8; ++ci) {
        int x = i + xcorner[ci];
        int y = j + ycorner[ci];
        int z = k + zcorner[ci];
        if (x < 0 || x >= _nsteps_x || y < 0 || y >= _nsteps_y || z < 0 || z >= _nsteps_z) {
            throw std::out_of_range("Cell corner indices are out of bounds.");
        }
        cell_values[ci] = _scalar_fields.at(name)[(z * _nsteps_y + y) * _nsteps_x + x];
    }
    return cell_values;
}
std::vector<Triangle> Grid::extractSurfaceForCell(int i, int j, int k, std::string name, double isovalue) const {
    if (i < 0 || i >= _nsteps_x || j < 0 || j >= _nsteps_y || k < 0 || k >= _nsteps_z) {
        throw std::out_of_range("Cell indices are out of bounds.");
    }

    std::array<double, 8> cell_values = get_cell_values(i,j,k,name); 
    
    std::vector<Triangle> triangle_soup = marchingCubes(cell_values,isovalue, _origin_x + i * _step_x, 
                                _origin_y + j * _step_y, 
                                _origin_z + k * _step_z, 
                                _step_x, _step_y, _step_z);

    // Create and return a TriangleMesh object representing the surface
    return triangle_soup;
}

NumpyMesh Grid::_extract_isosurface(std::string name, double isovalue) const {
    std::vector<Triangle> triangles;
    
    for (int i = 0; i < _nsteps_x - 1; ++i) {
        for (int j = 0; j < _nsteps_y - 1; ++j) {
            for (int k = 0; k < _nsteps_z - 1; ++k) {
                auto cell_triangles = extractSurfaceForCell(i, j, k, name, isovalue);
                triangles.insert(triangles.end(), cell_triangles.begin(), cell_triangles.end());
            }
        }
    }
    TriangleMesh surface;

    for (ssize_t i = 0; i < triangles.size(); ++i)
    {
    std::array<TriangleMesh::Vertex_index,3> vertex_indices;
        int j=0;
        for (const auto &point : triangles[i]) {
            // Add each point of the triangle to the surface mesh
            vertex_indices[j] = surface.add_vertex(point);
            j++;


        }
        surface.add_face(vertex_indices[0], vertex_indices[1], vertex_indices[2]);
    }
        
  PMP::autorefine(surface);
  std::vector<std::vector<int>> tri_indices;
for (auto f : surface.faces())
    {
        std::vector<int> tri;
        for (auto he : CGAL::halfedges_around_face(surface.halfedge(f), surface))
            tri.push_back(CGAL::target(he, surface));

        
        tri_indices.push_back(tri);
        
    }


    std::vector<std::vector<double>> vertices;
    for (const auto &v : surface.vertices()) {
        const auto &point = surface.point(v);
        vertices.push_back({point.x(), point.y(), point.z()});
    }
    pybind11::array_t<double> vertices_array(
        {static_cast<int>(vertices.size()), 3});
    auto vbuf = vertices_array.mutable_unchecked<2>();
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        vbuf(i, 0) = vertices[i][0];
        vbuf(i, 1) = vertices[i][1];
        vbuf(i, 2) = vertices[i][2];
    }

    pybind11::array_t<int> triangles_array(
        {static_cast<int>(tri_indices.size()), 3});
    auto tbuf = triangles_array.mutable_unchecked<2>();
    for (size_t i = 0; i < tri_indices.size(); ++i)
    {
        tbuf(i, 0) = tri_indices[i][0];
        tbuf(i, 1) = tri_indices[i][1];
        tbuf(i, 2) = tri_indices[i][2];
    }

    // —‑‑‑‑‑ 4.  Package & return ------------------------------------------
    NumpyMesh result;
    result.vertices = vertices_array;
    result.triangles = triangles_array;
    return result;
 
}

std::vector<bool> Grid::calculate_cell_mask(std::string scalar_name, bool greater, double value) const {
    std::vector<bool> mask;
    // for (int i =0; i<_nsteps_x; i++) {
    //     for (int j=0; j<_nsteps_y; j++){
    //         for (int k = 0; k<_nsteps_z; k++) {
    //             std::array<double,8> cell_values = get_cell_values(i,j,k,scalar_name);
    //             bool cell_mask = true;
    //             for (int n =0; n<8; n++)
    //             {
                    
    //                 if (greater)
    //                     cell_mask &= cell_values[n] > value;
    //                 else
    //                     cell_mask &= cell_values[n] < value;
                    
    //             }
    //             mask.push_back(cell_mask);


    //         }
    //     }

    // }
    return mask;

}

double Grid::step_vector_x() const{
return _step_x;
}
double Grid::step_vector_y() const{
return _step_y;
}
double Grid::step_vector_z() const{
return _step_z;
}
int Grid::nsteps_x() const{
    return _nsteps_x;

}
int Grid::nsteps_y() const{
    return _nsteps_y;
}
int Grid::nsteps_z() const{
    return _nsteps_z;
}