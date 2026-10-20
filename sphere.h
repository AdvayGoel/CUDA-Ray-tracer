#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h" 

class 
sphere : public hittable {
  public:
    SphereData data;
    gpu_material* mat;
    aabb bbox;

    __device__ sphere(const point3& center, float radius, gpu_material* mat) : mat(mat) {
        data.center = center;
        data.radius = fmaxf(1e-6f,radius);
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }

    __device__ __host__ aabb bounding_box() const override {
        return bbox;
    }

    __device__ PrimitiveType get_type() const override { return PrimitiveType::Sphere; }
};

#endif