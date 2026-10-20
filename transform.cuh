#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

#include "hittable.h"
#include "vec3.cuh"
#include "device_math.cuh"
#include "mat3.cuh"

class transform : public hittable {
public:
    hittable* object;
    vec3 offset;
    mat3 rotation;
    aabb bbox;
    gpu_material* mat;
    __host__ __device__ transform(hittable* object, const vec3& offset, float rx, float ry, float rz, gpu_material* material = nullptr)
        : object(object), offset(offset), mat(material) {
        mat3 mx = mat3::rotate_x(rx);
        mat3 my = mat3::rotate_y(ry);
        mat3 mz = mat3::rotate_z(rz);

        rotation = mz * my * mx;
        bbox = compute_bounding_box();
    }


    __host__ __device__ aabb bounding_box() const override {
        return bbox;
    }

    __device__ PrimitiveType get_type() const override { return PrimitiveType::Transform; }

private:

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