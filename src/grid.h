#ifndef GRID_H
#define GRID_H
#include "mesh.h"
#include "marchingcubes.h"
class Grid
{
public:
    // Constructor
    Grid(double origin_x, double origin_y, double origin_z, double step_x, double step_y, double step_z, int nsteps_x, int nsteps_y, int nsteps_z);
    void addScalarField(const std::string &name, const std::vector<double> &values);
    std::vector<Triangle> extractSurfaceForCell(int i, int j, int k, std::string name, double isovalue) const;
private:
    double _origin_x, _origin_y, _origin_z; // Origin of the grid
    double _step_x, _step_y, _step_z;       // Step sizes in each dimension
    int _nsteps_x, _nsteps_y, _nsteps_z;   // Number of steps in each dimension
    std::map<std::string, std::vector<double>> _scalar_fields; // Scalar fields associated with the grid
};
#endif // GRID_H