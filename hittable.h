#ifndef HITTABLE_H
#define HITTABLE_H

#include "device_math.cuh"
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

        __device__ virtual light_sample sample_light_point(unsigned int* local_rand_state) const {
            light_sample sample;
            sample.Le = colour(0.0f, 0.0f, 0.0f);
            sample.normal = vec3(0.0f, 0.0f, 0.0f);
            sample.pdf_area = 0;
            sample.position = point3(0, 0, 0);
            return sample;
        };

        __device__ virtual vec3 geometric_normal(const point3 at) const {
            return vec3(0.0f, 0.0f, 0.0f);
        }

        __device__ virtual float area_val() const {
            return 0;
        };
    };

#endif
