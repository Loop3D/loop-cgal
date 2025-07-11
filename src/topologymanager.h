#ifndef TOPOLOGYMANAGER_H
#define TOPOLOGYMANAGER_H
#include "grid.h"
#include "marchingcubes.h"
struct Node{
    std::string scalar_field_name;
    double isovalue;
};
class TopologyManager
{
public:
    // Constructor
    TopologyManager(Grid &grid);
    Node add_surface(std::string scalar_field_name, double isovalue);
    void add_edge(Node surface, Node clipper);
    std::pair<std::string, TriMesh> getSurfaces() const;
private:
    Grid &_grid;
    std::vector<Node> _surfaces;
    std::pair<Node, Node> _topology;
};
#endif // GRID_H