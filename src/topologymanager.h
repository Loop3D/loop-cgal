#ifndef TOPOLOGYMANAGER_H
#define TOPOLOGYMANAGER_H
#include "grid.h"
#include "marchingcubes.h"
class Node{
    public:
        Node(std::string scalar_field_name, double isovalue);
        std::string name();
        double isovalue();
        std::string scalar_field_name();
    private:
        std::string _scalar_field_name;
        double _isovalue;
        std::string _name;
};
class TopologyManager
{
public:
    // Constructor
    TopologyManager(Grid &grid);
    Node* add_surface(std::string scalar_field_name, double isovalue);
    void add_edge(Node* surface, Node* clipper);
    NumpyMesh get_surface(std::string surface_name);
    std::vector<std::string> get_surface_names();
private:
    Grid &_grid;
    std::map<std::string, Node*> _surfaces;
    std::map<std::string, std::vector<Node*>> _topology;
};
#endif // GRID_H