#include "topologymanager.h"
#include <sstream>
Node::Node(std::string scalar_field_name, double isovalue) : _scalar_field_name(scalar_field_name),_isovalue(isovalue) {
    std::stringstream ss;
    ss <<_scalar_field_name<<"_"<<_isovalue;
    _name = ss.str();
}
double Node::isovalue(){
    return _isovalue;
}
std::string Node::scalar_field_name(){
    return _scalar_field_name;
}
TopologyManager::TopologyManager(Grid &grid) : _grid(grid) {}

Node* TopologyManager::add_surface(std::string scalar_field_name, double isovalue) {
    //store a referene to the surface node
    Node* surface = new Node(scalar_field_name, isovalue);
    _surfaces[surface->name()] = surface;
    return surface;


   
}

void TopologyManager::add_edge(Node* surface, Node* clipper) {
    _topology[surface->name()].push_back(clipper);
    return;

}

NumpyMesh TopologyManager::get_surface(std::string name){ 
    std::vector<Node*> clippers = _topology[name];
    Node *surface = _surfaces[name];
    std::vector<std::vector<bool>> masks;
    const int n_surfaces = get_surface_names().size();
    std::vector<std::array<double, 8>> cell_values(n_surfaces);
    for (int i = 0; i < _grid.nsteps_x(); i++){
        for (int j=0; j<_grid.nsteps_y(); j++){
            for (int k=0; k<_grid.nsteps_z(); k++)
                {
                    
                    //check if surface should exist in this cell

                    //if it does calculate the isosurface

                    //check if any of the clippers exist

                    
                }
        }
    }
    // for (int i = 0; i < clippers.size(); i++){
    //     masks.push_back(grid.calculate_cell_mask(clippers[i]->_scalar_field_name,clippers[i]->_isovalue))
    

    // }
    // std::vector<bool> mask;
    // if (masks.size() == 1) {
    //     mask = masks[0];
    // }
    //else combine masks

    NumpyMesh mesh;
    return mesh;


}