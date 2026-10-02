#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

#include "hittable.h"
#include "vec3.cuh"
#include "device_math.cuh"
#include "mat3.cuh"

class transform : public hittable {
public:
    __host__ __device__ transform(hittable* object, const vec3& offset, float rx, float ry, float rz, material* material = nullptr)
        : object(object), offset(offset), mat(material) {
        mat3 mx = mat3::rotate_x(rx);
        mat3 my = mat3::rotate_y(ry);
        mat3 mz = mat3::rotate_z(rz);

        rotation = mz * my * mx;
        bbox = compute_bounding_box();
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // For pure rotation, inverse is transpose: R^T * (p - offset)
        mat3 inv_rot = rotation.transpose();
        ray r_local(inv_rot * (r.origin() - offset), inv_rot * r.direction());

        if (!object->hit(r_local, ray_t, rec))
            return false;

        // t remains valid without scale adjustment
        rec.p = r.at(rec.t);
        rec.normal = rotation * rec.normal;

        if (mat != nullptr) {
            rec.mat = mat;
        }

        return true;
    }

    __host__ __device__ aabb bounding_box() const override {
        return bbox;
    }

    __device__ light_sample sample_light_point(unsigned int *local_rand_state) const override {
        light_sample sample = object->sample_light_point(local_rand_state);
        if (mat != nullptr) {
            sample.Le = mat->emitted();
        }
        sample.normal = rotation * sample.normal;
        sample.position = rotation * sample.position + offset;
        return sample;
    }

    __device__ vec3 geometric_normal(const point3 at) const override {
        vec3 local_norm = object->geometric_normal(rotation.transpose() * (at - offset));
        return rotation * local_norm;

    }

private:
    hittable* object;
    vec3 offset;
    mat3 rotation;
    aabb bbox;
    material* mat;

    __host__ __device__ aabb compute_bounding_box() const {
        aabb box = object->bounding_box();
        point3 local_center = 0.5f * point3(box.x.min + box.x.max, box.y.min + box.y.max, box.z.min + box.z.max);
        vec3 local_extents  = 0.5f * vec3(box.x.size(), box.y.size(), box.z.size());

        point3 world_center = (rotation * local_center) + offset;

        // Projected world extents using absolute values of rotation matrix elements
        vec3 world_extents(
            fabsf(rotation(0, 0)) * local_extents.x() + fabsf(rotation(0, 1)) * local_extents.y() + fabsf(rotation(0, 2)) * local_extents.z(),
            fabsf(rotation(1, 0)) * local_extents.x() + fabsf(rotation(1, 1)) * local_extents.y() + fabsf(rotation(1, 2)) * local_extents.z(),
            fabsf(rotation(2, 0)) * local_extents.x() + fabsf(rotation(2, 1)) * local_extents.y() + fabsf(rotation(2, 2)) * local_extents.z()
        );

        return aabb(world_center - world_extents, world_center + world_extents);
    }
};

#endif