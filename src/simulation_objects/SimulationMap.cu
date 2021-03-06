#include <cmath>
#include <utility>

#include "SimulationMap.cuh"
#include "geometric/Polyhedron.cuh"
#include "geometric/Face.cuh"
#include "../jones_constants.hpp"
#include "../external/random_generator.cuh"

namespace jc = jones_constants;


__device__ double const mapnode_dist = 2 * jc::speed;


__device__ SimulationMap::SimulationMap(Polyhedron *polyhedron) :
        polyhedron(polyhedron)
{
    typedef bool(MapNode::*SetNodeMethod)(MapNode *);
    typedef MapNode *(MapNode::*GetNodeMethod)() const;


    /**
     * Maximum number of nodes
     * It must be greater than or equal to the real number of nodes (`n_of_nodes`) after the node grid creation
     */
    const int max_number_of_nodes = 2 * polyhedron->calculate_square_of_surface() / (mapnode_dist * mapnode_dist);

    Face *start_face = &polyhedron->get_faces()[0];
    SpacePoint start_node_coordinates = (start_face->get_vertices()[0] + start_face->get_vertices()[1] +
                                         start_face->get_vertices()[2]) / 3;

    nodes = new MapNode[max_number_of_nodes];
    nodes[0] = MapNode(polyhedron, 0, start_node_coordinates); // `0` is the index of `start_face`
    n_of_nodes = 1;

    // Direction vector of the first node to the top neighbor
    SpacePoint direction_vector = relative_point_rotation(start_node_coordinates, start_face->get_vertices()[0],
                                                          start_face->get_normal(), M_PI * 2 * rand0to1()) -
                                  start_node_coordinates;

    /**
     * Array of direction vectors to top neighbors for each polyhedron face
     * `top_neighbor_directions_for_faces[i]` corresponds to `polyhedron->get_faces()[i]` (i. e. indexes are same)
     */
    auto *top_neighbor_directions_for_faces = new SpacePoint[polyhedron->get_n_of_faces()];
    top_neighbor_directions_for_faces[0] = direction_vector * mapnode_dist / get_distance(direction_vector, origin);

    /**
     * Boolean array where i-th element tells whether the i-th face have nodes or not
     * `does_face_have_nodes[i]` value corresponds to `polyhedron->faces[i]` face
     */
    auto *does_face_have_nodes = new bool[polyhedron->get_n_of_faces()];
    does_face_have_nodes[0] = true;
    for(int i = 1; i < polyhedron->get_n_of_faces(); ++i)
        does_face_have_nodes[i] = false;

    /**
     * Array of pointers to the `MapNode` member functions
     * Each of them returns the particular neighbor node
     * First array element corresponds to a top neighbor,
     * the following elements correspond to the following neighbors counterclockwise
     */
    GetNodeMethod get_node_neighbors[] = {
            &MapNode::get_top,
            &MapNode::get_left,
            &MapNode::get_bottom,
            &MapNode::get_right
    };

    /**
     * Array of pointers to the `MapNode` member functions
     * Each of them sets the link from current node to the particular neighbor node
     * First array element corresponds to a top neighbor,
     * the following elements correspond to the following neighbors counterclockwise
     */
    SetNodeMethod set_node_neighbors[] = {
            &MapNode::set_top,
            &MapNode::set_left,
            &MapNode::set_bottom,
            &MapNode::set_right
    };

    bool create_new_nodes = true;  // New nodes are allowed to be created

    // Creating new nodes while it is possible, some nodes may have less neighbors than four
    for(int current_node_id = 0; current_node_id < n_of_nodes; ++current_node_id)
    {
        MapNode &current_node = nodes[current_node_id];
        double angle = 0;
        for(int i = 0; i < 4; ++i)
        {
            if((current_node.*get_node_neighbors[i])() == nullptr)
            {
                int neighbor_node_id = get_neighbor_node_id(current_node_id, top_neighbor_directions_for_faces, angle,
                                                            does_face_have_nodes, create_new_nodes);
                if(neighbor_node_id != -1)
                {
                    (current_node.*set_node_neighbors[i])(&nodes[neighbor_node_id]);
                }
            }
            angle += M_PI_2;
        }

        if(!create_new_nodes and current_node.get_face()->get_node() == nullptr)
        {
            current_node.get_face()->set_node(&current_node, polyhedron);
        }

        if(create_new_nodes && current_node_id == n_of_nodes - 1)
        {
            /*
             * Starting from this moment: setting all pointers to neighbors that were not set earlier
             * The node that will be set as the neighbor is the closest node to hypothetical coordinates of neighbor
             */

            // Returns to the beginning of the `SimulationMap::nodes` array
            current_node_id = -1;

            // All nodes were created
            create_new_nodes = false;
        }
    }

    delete[] top_neighbor_directions_for_faces;
    delete[] does_face_have_nodes;
}

__host__ __device__ SimulationMap &SimulationMap::operator=(SimulationMap &&other) noexcept
{
    if(this != &other)
    {
        swap(n_of_nodes, other.n_of_nodes);
        swap(nodes, other.nodes);
        swap(polyhedron, other.polyhedron);
    }

    return *this;
}

__host__ __device__ SimulationMap::SimulationMap(SimulationMap &&other) noexcept
{
    nodes = nullptr;

    *this = std::move(other);
}

__host__ __device__ SimulationMap::SimulationMap()
{
    _reset_destructively();
}

__device__ SimulationMap::~SimulationMap()
{
    delete[] nodes;
}


__host__ __device__ void SimulationMap::_reset_destructively()
{
    nodes = nullptr;
}


__device__ int SimulationMap::get_n_of_nodes() const
{
    return this->n_of_nodes;
}

__global__ void get_n_of_nodes(const SimulationMap *const simulation_map, int *return_value)
{
    STOP_ALL_THREADS_EXCEPT_FIRST;

    *return_value = simulation_map->get_n_of_nodes();
}


__device__ long SimulationMap::find_face_index(Face *face) const
{
    long index = face - &polyhedron->get_faces()[0];
    if(0 <= index && index < polyhedron->get_n_of_faces())
        return index;
    return -1;
}


__device__ SpacePoint SimulationMap::calculate_neighbor_node_coordinates(int current_node_id, SpacePoint top_direction,
                                                                         double angle, bool do_projection) const
{
    MapNode &current_node = nodes[current_node_id];
    Face *current_face = current_node.get_face();
    SpacePoint neighbor_coordinates = relative_point_rotation(current_node.get_coordinates(),
                                                              current_node.get_coordinates() + top_direction,
                                                              current_face->get_normal(),
                                                              angle);
    if(do_projection)
    {
        neighbor_coordinates = get_projected_vector_end(current_node.get_coordinates(), neighbor_coordinates,
                                                        current_face, polyhedron);
    }
    return neighbor_coordinates;
}


__device__ int SimulationMap::find_index_of_nearest_node(SpacePoint dest) const
{
    int nearest_mapnode_id = 0;
    double best_distance = get_distance(nodes[0].get_coordinates(), dest);

    for(int neighbor = 2; neighbor < n_of_nodes; ++neighbor)
    {
        double current_distance = get_distance(nodes[neighbor].get_coordinates(), dest);
        if(current_distance < best_distance)
        {
            nearest_mapnode_id = neighbor;
            best_distance = current_distance;
        }
    }
    return nearest_mapnode_id;
}


__device__ void SimulationMap::set_direction_to_top_neighbor(int current_node_id, int neighbor_node_id,
                                                             SpacePoint *top_neighbor_directions_for_faces,
                                                             double angle) const
{
    MapNode &current_node = nodes[current_node_id];
    MapNode &neighbor_node = nodes[neighbor_node_id];

    if(neighbor_node.get_face() != current_node.get_face())
    {
        SpacePoint new_direction = neighbor_node.get_coordinates() -
                                   find_intersection_with_edge(
                                           current_node.get_coordinates(),
                                           calculate_neighbor_node_coordinates(
                                                   current_node_id,
                                                   top_neighbor_directions_for_faces[current_node.get_face_index()],
                                                   angle, false),
                                           current_node.get_face());
        new_direction = relative_point_rotation(neighbor_node.get_coordinates(),
                                                neighbor_node.get_coordinates() + new_direction,
                                                neighbor_node.get_face()->get_normal(),
                                                -angle) -
                        neighbor_node.get_coordinates();
        top_neighbor_directions_for_faces[neighbor_node.get_face_index()] =
                new_direction * mapnode_dist / get_distance(new_direction, origin);
    }
}


__device__ int SimulationMap::get_neighbor_node_id(int current_node_id, SpacePoint *top_neighbor_directions_for_faces,
                                                   double angle, bool *does_face_have_nodes, bool create_new_nodes)
{
    Face *current_face = nodes[current_node_id].get_face();
    int current_face_index = nodes[current_node_id].get_face_index();

    // Hypothetical coordinates of neighbor node counted using direction to the top neighbor and `angle`
    SpacePoint neighbor_coordinates = calculate_neighbor_node_coordinates(
            current_node_id, top_neighbor_directions_for_faces[current_face_index], angle, true);

    Face *next_face = polyhedron->find_face_by_point(neighbor_coordinates);
    long next_face_index = find_face_index(next_face);

    int nearest_node_id = find_index_of_nearest_node(neighbor_coordinates);
    if(!create_new_nodes || (next_face == nodes[nearest_node_id].get_face() &&
                             get_distance(nodes[nearest_node_id].get_coordinates(), neighbor_coordinates) < eps))
    {
        // Neighbor node has already existed
        return nearest_node_id;
    }
    else if(current_face == next_face || !does_face_have_nodes[next_face_index])
    {
        // Neighbor node does not exist, but it can be created

        nodes[n_of_nodes] = MapNode(polyhedron, next_face_index, neighbor_coordinates);

        set_direction_to_top_neighbor(current_node_id, n_of_nodes, top_neighbor_directions_for_faces, angle);

        does_face_have_nodes[next_face_index] = true;

        return n_of_nodes++;
    }
    return -1;
}
