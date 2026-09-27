#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.cuh"
#include "aabb.cuh"

class material; // resolves circular referencing issues by forward declaration
class hittable;

class hit_record {
    public:
        point3 p;
        vec3 normal;
        material* mat;
        float t;
        float u;
        float v;
        bool front_face;
        const hittable* object = nullptr;

        // sets the normal to the opposite direction to the ray direction.
        HD void set_face_normal(const ray& r, const vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0.0f;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

class hittable {
    public:
        
        __device__ __host__ virtual ~hittable() = default; // destructor

        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

        __device__ __host__ virtual aabb bounding_box() const = 0;
};

#endif
