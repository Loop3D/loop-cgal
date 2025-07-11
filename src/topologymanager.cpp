#include "topologymanager.h"

TopologyManager::TopologyManager(Grid &grid) : _grid(grid) {}

Node TopologyManager::add_surface(std::string scalar_field_name, double isovalue) {
    Node surface;
    _surfaces.push_back(surface);
    return surface
}

void TopologyManager::add_edge(Node surface, Node clipper) {
    return

}