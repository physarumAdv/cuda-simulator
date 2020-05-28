#include "Particle.cuh"
#include "Face.cuh"
#include "fucking_shit.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__device__ Particle::Particle(MapNode *map_node, SpacePoint coordinates, double angle) :
    coordinates(coordinates), map_node(map_node)
{
    Face current_face = map_node->polyhedron->faces[map_node->polyhedron_face_id];
    this->normal = current_face.normal;

    SpacePoint radius = current_face.vertices[0] - coordinates;
    radius = radius * jc::so / get_distance(radius, origin);
    this->middle_sensor = this->rotate_point_angle(radius, angle);
    this->init_left_right_sensors();
}


__device__ SpacePoint Particle::rotate_point_angle(SpacePoint radius, double angle) const
{
    double angle_cos = cos(angle);
    return (1 - angle_cos) * (this->normal * radius) * this->normal + angle_cos * radius +
                      sin(angle) * (this->normal % radius) + this->coordinates;
}

__device__ void Particle::init_left_right_sensors()
{
    this->left_sensor = this->rotate_point_angle(this->middle_sensor - this->coordinates, jc::sa);
    this->right_sensor = this->rotate_point_angle(this->middle_sensor - this->coordinates, -jc::sa);
    if(this->normal * ((this->right_sensor - this->coordinates) % (this->left_sensor - this->coordinates)) < 0)
    {
        SpacePoint p = this->right_sensor;
        this->right_sensor = this->left_sensor;
        this->left_sensor = p;
    }
}


__device__ void Particle::do_sensory_behaviours()
{
    double trail_l = find_nearest_mapnode(map_node->polyhedron, left_sensor, map_node)->trail;
    double trail_m = find_nearest_mapnode(map_node->polyhedron, middle_sensor, map_node)->trail;
    double trail_r = find_nearest_mapnode(map_node->polyhedron, right_sensor, map_node)->trail;

    if((trail_m > trail_l) && (trail_m > trail_r)) // m > l, r
        return;
    if((trail_m < trail_l) && (trail_m < trail_r)) // m < l, r
    {
        if(trail_l < trail_r) // m < l < r
            rotate(jc::ra);
        else // m < r <= l
            rotate(-jc::ra);

        return;
    }
    if(trail_l < trail_r) // l < m < r
        rotate(jc::ra);
    else // r < m < l
        rotate(-jc::ra);
}

__device__ void Particle::rotate(double angle)
{
    SpacePoint c = coordinates;
    Polyhedron *p = map_node->polyhedron;
    int f = map_node->polyhedron_face_id;

    left_sensor = get_projected_vector_end(c, rotate_point_angle(left_sensor, angle), f, p);
    middle_sensor = get_projected_vector_end(c, rotate_point_angle(middle_sensor, angle), f, p);
    right_sensor = get_projected_vector_end(c, rotate_point_angle(right_sensor, angle), f, p);
}
