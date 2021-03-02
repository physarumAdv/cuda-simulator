#ifdef COMPILE_FOR_CPU

#include <cmath>

#endif //COMPILE_FOR_CPU

#include <cstdio>

#include "simulation_logic.cuh"
#include "../external/random_generator.cuh"
#include "../simulation_objects/Particle.cuh"
#include "../jones_constants.hpp"

namespace jc = jones_constants;


[[nodiscard]] __device__ bool create_particle(MapNode *node)
{
    auto p = new Particle(node, rand0to1() * 2 * M_PI);

    if(node->attach_particle(p))
        return true;

    delete p;
    return false;
}

[[nodiscard]] __device__ bool delete_particle(MapNode *node)
{
    Particle *p = node->get_particle();

    if(!node->detach_particle(p))
        return false;

    delete p;
    return true;
}


__host__ __device__ int count_particles_in_node_window(MapNode *node, int window_size)
{
    for(int i = 0; i < window_size / 2; ++i)
        node = node->get_top()->get_left();

    MapNode *row = node;
    int ans = 0;
    for(int i = 0; i < window_size; ++i)
    {
        MapNode *cur = row;
        for(int j = 0; j < window_size; ++j)
        {
            if(cur->does_contain_particle())
                ++ans;
            cur = cur->get_right();
        }
        row = row->get_bottom();
    }

    return ans;
}


__device__ bool random_death_test(MapNode *node)
{
    if(rand0to1() < jc::random_death_probability)
    {
        if(!delete_particle(node))
        {
            // This is what called "undefined behaviour" in the docs :)
            printf("%s:%d - this line should never be reached", __FILE__, __LINE__);
            return false; // Particle was not removed
        }
        return true; // Particle was removed
    }
    return false; // Particle was not removed
}

__device__ bool death_test(MapNode *node)
{
    int particles_in_window = count_particles_in_node_window(node, jc::sw);
    if(jc::smin <= particles_in_window && particles_in_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
    {
        if(!delete_particle(node))
        {
            // This is what called "undefined behaviour" in the docs :)
            printf("%s:%d - this line should never be reached", __FILE__, __LINE__);
            return false; // Particle was not removed
        }
        return true; // Particle was removed
    }
    return false; // Particle was not removed
}

__device__ void division_test(MapNode *node)
{
    int particle_window = count_particles_in_node_window(node, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
    {
        if(rand0to1() <= jc::division_probability)
        {
            MapNode *row = node->get_top()->get_left();
            for(int i = 0; i < 3; ++i)
            {
                MapNode *cur = row;
                for(int j = 0; j < 3; ++j)
                {
                    if(create_particle(cur)) // If new particle was successfully created
                        return;
                    cur = cur->get_right();
                }
                row = row->get_bottom();
            }
        }
    }
}


__host__ __device__ Polyhedron generate_cube(double edge_length)
{
    SpacePoint vertices1[] = {
            {0,           0, 0},
            {0,           0, edge_length},
            {edge_length, 0, edge_length},
            {edge_length, 0, 0}
    };
    SpacePoint vertices2[] = {
            {0, 0,           0},
            {0, edge_length, 0},
            {0, edge_length, edge_length},
            {0, 0,           edge_length}
    };
    SpacePoint vertices3[] = {
            {0,           0,           0},
            {edge_length, 0,           0},
            {edge_length, edge_length, 0},
            {0,           edge_length, 0}
    };
    SpacePoint vertices4[] = {
            {edge_length, 0,           edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, edge_length, 0},
            {edge_length, 0,           0}
    };
    SpacePoint vertices5[] = {
            {0,           0,           edge_length},
            {0,           edge_length, edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, 0,           edge_length}
    };
    SpacePoint vertices6[] = {
            {edge_length, edge_length, 0},
            {edge_length, edge_length, edge_length},
            {0,           edge_length, edge_length},
            {0,           edge_length, 0}
    };

    Face faces[] = {
            Face(vertices1, 4),
            Face(vertices2, 4),
            Face(vertices3, 4),
            Face(vertices4, 4),
            Face(vertices5, 4),
            Face(vertices6, 4)
    };

    Polyhedron cube(std::move(faces), 6);
    return cube;
}
