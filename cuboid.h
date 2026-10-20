#ifndef CUBOID_H
#define CUBOID_H

#include "hittable.h"
#include "vec3.cuh"
#include "device_math.cuh"
#include "mat3.cuh"

class
cuboid : public hittable {
    public:
        CuboidData data;
        gpu_material* mat;
        aabb bbox;

        __device__ aabb compute_bounding_box() const {
            // Local center and half-extents
            point3 local_center = 0.5f * (data.p_min + data.p_max);
            vec3 local_extents  = 0.5f * (data.p_max - data.p_min);

            // World center
            point3 world_center = (data.rotation * local_center) + data.center;
            mat3 rotation = data.rotation;
            // Projected world extents using absolute values of rotation matrix elements
            vec3 world_extents(
                fabsf(rotation(0, 0)) * local_extents.x() + fabsf(rotation(0, 1)) * local_extents.y() + fabsf(rotation(0, 2)) * local_extents.z(),
                fabsf(rotation(1, 0)) * local_extents.x() + fabsf(rotation(1, 1)) * local_extents.y() + fabsf(rotation(1, 2)) * local_extents.z(),
                fabsf(rotation(2, 0)) * local_extents.x() + fabsf(rotation(2, 1)) * local_extents.y() + fabsf(rotation(2, 2)) * local_extents.z()
            );
            return aabb(world_center - world_extents, world_center + world_extents);
        }

        __device__ void init_area() {
            vec3 size = vec3(data.p_max.x() - data.p_min.x(), data.p_max.y() - data.p_min.y(), data.p_max.z() - data.p_min.z());
            float ax = size.y() * size.z();
            float ay = size.x() * size.z();
            float az = size.x() * size.y();
            data.area = 2.0f * (ax + ay + az);
        }

        __device__ cuboid(const point3& pivot, const point3& p_min, const point3& p_max, gpu_material* mat, float rx = 0, float ry = 0, float rz = 0) : mat(mat) {
            data.center = pivot;
            data.p_min = p_min;
            data.p_max = p_max;

            mat3 mx = mat3::rotate_x(rx);
            mat3 my = mat3::rotate_y(ry);
            mat3 mz = mat3::rotate_z(rz);

            data.rotation = mz * my * mx;
            data.inverse_rotation = data.rotation.transpose();

            bbox = compute_bounding_box();
            init_area();
        }

        __device__ cuboid(const point3& center, float width, float height, float depth, gpu_material* mat, float rx = 0, float ry = 0, float rz = 0) : mat(mat) {
            vec3 half_size = 0.5f * vec3(width, height, depth);
            data.p_min = -half_size;
            data.p_max = half_size;
            data.center = center;
            mat3 mx = mat3::rotate_x(rx);
            mat3 my = mat3::rotate_y(ry);
            mat3 mz = mat3::rotate_z(rz);

            data.rotation = mz * my * mx;
            data.inverse_rotation = data.rotation.transpose();

            bbox = compute_bounding_box();
            init_area();

        }

        __device__ __host__ aabb bounding_box() const override {
            return bbox;
        }

        __device__ PrimitiveType get_type() const override { return PrimitiveType::Cuboid; }
};  

#endif