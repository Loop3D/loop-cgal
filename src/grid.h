#ifndef GRID_H
#define GRID_H
#include "mesh.h"
#include "marchingcubes.h"
#include "numpymesh.h"

class Grid
{
public:
    // Constructor
    Grid(double origin_x, double origin_y, double origin_z, double step_x, double step_y, double step_z, int nsteps_x, int nsteps_y, int nsteps_z);
    void addScalarField(const std::string &name, const std::vector<double> &values);
    std::vector<Triangle> extractSurfaceForCell(int i, int j, int k, std::string name, double isovalue) const;
    NumpyMesh _extract_isosurface(std::string name, double isovalue) const;
    std::array<double, 8> get_cell_values(int i, int j, int k, std::string name) const;
    std::vector<bool> calculate_cell_mask(std::string scalar_name, bool greater, double value) const;
    double step_vector_x() const;
    double step_vector_y() const;
    double step_vector_z() const;
    int nsteps_x() const;
    int nsteps_y() const;
    int nsteps_z() const;
    std::array<int,3> nsteps() const;
private:
    double _origin_x, _origin_y, _origin_z; // Origin of the grid
    double _step_x, _step_y, _step_z;       // Step sizes in each dimension
    int _nsteps_x, _nsteps_y, _nsteps_z;   // Number of steps in each dimension
    std::map<std::string, std::vector<double>> _scalar_fields; // Scalar fields associated with the grid
};
#endif // GRID_H