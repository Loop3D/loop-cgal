#include "grid.h"

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

std::vector<Triangle> Grid::extractSurfaceForCell(int i, int j, int k, std::string name, double isovalue) const {
    if (i < 0 || i >= _nsteps_x || j < 0 || j >= _nsteps_y || k < 0 || k >= _nsteps_z) {
        throw std::out_of_range("Cell indices are out of bounds.");
    }

    // Extract the scalar field values for the specified cell
    std::array<double,8> cell_values;
    int ci = 0;
    for (int z = 0; z < _nsteps_z; ++z) {
        for (int y = 0; y < _nsteps_y; ++y) {
            for (int x = 0; x < _nsteps_x; ++x) {
                cell_values[ci] = _scalar_fields.at(name)[(z * _nsteps_y + y) * _nsteps_x + x];
            }
        }
    }
    std::vector<Triangle> triangle_soup = marchingCubes(cell_values,isovalue, _origin_x + i * _step_x, 
                                _origin_y + j * _step_y, 
                                _origin_z + k * _step_z, 
                                _step_x, _step_y, _step_z);

    // Create and return a TriangleMesh object representing the surface
    return triangle_soup;
}

