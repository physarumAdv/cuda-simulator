#ifndef MINDS_CRAWL_SIMULATION_LOGIC_CUH
#define MINDS_CRAWL_SIMULATION_LOGIC_CUH


#include "../simulation_objects/MapNode.cuh"
#include "../simulation_objects/SimulationMap.cuh"
#include "../common.cuh"


/**
 * Creates a particle in the given node
 *
 * Creates a new `Particle` object with coordinates of the given node and attaches particle to it
 *
 * @param node Pointer to the node to create particle at
 *
 * @returns `true` if the `MapNode` was not occupied (which means a new `Particle` was created there), otherwise `false`
 *
 * @note This function does allocate new memory for a new `Particle`, but this memory is freed immediately if the
 *      created `Particle` was not attached to the `MapNode`
 */
[[nodiscard]] __device__ bool create_particle(MapNode *node);

/**
 * Removes a particle from the given node
 *
 * @param node Pointer to the node to remove particle from
 *
 * @returns `true`, if successfully deleted particle (which means there was a particle attached to the node and it
 *      wasn't replace by another one during deleting process), otherwise `false`
 *
 * @warning This operation is thread safe and cannot cause memory leaks, however it's highly recommended to run it only
 *      when there's no way for the particle to be updated by other threads
 */
[[nodiscard]] __device__ bool delete_particle(MapNode *node);

/**
 * Returns number of particles in a node window around the given node
 *
 * Let's assume a node of the model is a plane point with coordinates (X, Y). Then it's neighbors (even if they are not
 * equidistant) will have (X-1, Y-1), (X-1, Y), (X-1, Y+1), (X, Y-1), us, (X, Y+1), (X+1, Y-1), (X+1, Y), (X+1, Y+1).
 * Obviously, we can extrapolate this to neighbors of neighbors and so on. Then a node window around a node
 * with (X, Y) "coordinates" of size W (odd integer) is a set of nodes in the square with side W and central
 * point (X, Y). More formally a node window around (X, Y) node of size W is a set of nodes in a square with opposite
 * vertices (X - (W-1)/2, Y - (W-1)/2) and (X + (W-1)/2, Y + (W-1)/2). <br><br>
 * This function returns a number of particles in a node window around the given node of the given size
 *
 * @param node Pointer to the node to get node window around
 * @param window_size The node window size
 *
 * @note `window_size` must be a positive odd integer
 *
 * @returns The number of particles in the node window
 */
__host__ __device__ int count_particles_in_node_window(MapNode *node, int window_size);

/**
 * Runs a random death test in the given node
 *
 * Removes a particle in the given node with the `jones_constants::random_death_probability` probability
 *
 * @param node Pointer to the node to run random death test in
 *
 * @returns `true` if a particle was deleted from the node, otherwise `false`
 *
 * @note Return value does not tell you whether the operation was successful. It tells you whether the particle was
 *      removed, that's it
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ bool random_death_test(MapNode *node);

/**
 * Runs a death test in the given node
 *
 * Let's assume P is a number of particles in a node window around the given node of size `jones_constants::sw`. If
 * `jones_constants::smin` <= P <= `jones_constants::smax`, then nothing happens. Otherwise the particle is removed
 * from the given node
 *
 * @param node Pointer to the node to run death test in
 *
 * @returns `true` if a particle was deleted from the node, otherwise `false`
 *
 * @note Return value does not tell you whether the operation was successful. It tells you whether the particle was
 *      removed, that's it
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ bool death_test(MapNode *node);

/**
 * Runs a division test in the given node
 *
 * Let's assume P is a number of particles in a node window around the given node of size `jones_constants::gw`. If
 * `jones_constants::gmin` <= P <= `jones_constants::gmax` expression is FALSE, nothing happens. Otherwise (if the
 * expression is true) the division process begins with the `jones_constants::division_probability` probability. In the
 * division process we try to find a node without particle in a 3x3 node window around the given node. If we succeed,
 * we create a particle in the found node. Otherwise nothing happens
 *
 * @param node Pointer to the node to run division test in
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ void division_test(MapNode *node);


/**
 * Creates a cube and returns it
 *
 * Will be only used in the early stages of development, later will be replaced with a universal function building
 * arbitrary polyhedrons
 *
 * @param edge_length Length of the cube's edge
 *
 * @returns Cube represented wth a `Polyhedron` object
 */
__host__ __device__ Polyhedron generate_cube(double edge_length = 200);


#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
