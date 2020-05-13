#include "Particle.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;
SpacePoint origin = {0, 0, 0};

__device__ Particle::Particle(const Polyhedron *const polyhedron, int polyhedron_face,
                              SpacePoint coordinates, int angle)
{
    this->coordinates = coordinates;
    this->polyhedron_face = polyhedron_face;

    Face current_face = polyhedron.get_face(polyhedron_face);
    this->normal = current_face.normal;

    SpacePoint radius = polyhedron.get_vertex(current_face.vertices[0]) - coordinates;
    radius = radius * jc::so / get_distance(radius, origin);
    this->miggle_sensor = this.rotate_point_angle(radius, angle);
}

__device__ SpacePoint Particle::rotate_point_angle(SpacePoint radius, int angle)
{
    double angle_cos = cos(angle);
    return (1 - angle_cos) * (this->normal * radius) * this->normal + angle_cos * radius +
                      sin(angle) * (this->normal % radius) + this->coordinates;
}

__device__ void Particle::init_left_right_sensors()
{
    this->left_sensor = this.rotate_point_angle(this->middle_sensor - this->coordinates, jc::sa);
    this->right_sensor = this.rotate_point_angle(this->middle_sensor - this->coordinates, -jc::sa);
    if(this->normal * ((this->right_sensor - this->coordinates) % (this->left_sensor - this->coordinates)) < 0)
    {
        SpacePoint p = this->right_sensor;
        this->right_sensor = this->left_sensor;
        this->left_sensor = p;
    }
}