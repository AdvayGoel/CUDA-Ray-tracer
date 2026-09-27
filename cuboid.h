#ifndef CUBOID_H
#define CUBOID_H

#include "hittable.h"
#include "vec3.cuh"
#include "rtweekend.cuh"
#include "mat3.cuh"

class
cuboid : public hittable {
    public:
    point3 center;
    point3 p_min;
    point3 p_max;
    material* mat;
    mat3 rotation;
    mat3 inverse_rotation;
    aabb bbox;

    __device__ aabb compute_bounding_box() const {
        // Local center and half-extents
        point3 local_center = 0.5f * (p_min + p_max);
        vec3 local_extents  = 0.5f * (p_max - p_min);

        // World center
        point3 world_center = (rotation * local_center) + center;

        // Projected world extents using absolute values of rotation matrix elements
        vec3 world_extents(
            fabsf(rotation(0, 0)) * local_extents.x() + fabsf(rotation(0, 1)) * local_extents.y() + fabsf(rotation(0, 2)) * local_extents.z(),
            fabsf(rotation(1, 0)) * local_extents.x() + fabsf(rotation(1, 1)) * local_extents.y() + fabsf(rotation(1, 2)) * local_extents.z(),
            fabsf(rotation(2, 0)) * local_extents.x() + fabsf(rotation(2, 1)) * local_extents.y() + fabsf(rotation(2, 2)) * local_extents.z()
        );

        return aabb(world_center - world_extents, world_center + world_extents);
    }

    __device__ cuboid(const point3& pivot, const point3& p_min, const point3& p_max, material* mat, float rx = 0, float ry = 0, float rz = 0) : center(pivot), p_min(p_min), p_max(p_max), mat(mat) {
        mat3 mx = mat3::rotate_x(rx);
        mat3 my = mat3::rotate_y(ry);
        mat3 mz = mat3::rotate_z(rz);

        rotation = mz * my * mx;
        inverse_rotation = rotation.transpose();

        bbox = compute_bounding_box();
    }

    __device__ cuboid(const point3& center, float width, float height, float depth, material* mat, float rx = 0, float ry = 0, float rz = 0) : center(center), mat(mat) {
        vec3 half_size = 0.5f * vec3(width, height, depth);
        p_min = -half_size;
        p_max = half_size;

        mat3 mx = mat3::rotate_x(rx);
        mat3 my = mat3::rotate_y(ry);
        mat3 mz = mat3::rotate_z(rz);

        rotation = mz * my * mx;
        inverse_rotation = rotation.transpose();

        bbox = compute_bounding_box();

    }

    // Find algorithm in Notes
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {

        ray r_local = ray(inverse_rotation * (r.origin() - center), inverse_rotation * r.direction());

        // Pre-calculate the reciprocal of the direction to avoid repeated floating point division
        float inv_dx = 1.0f / r_local.direction().x();
        float inv_dy = 1.0f / r_local.direction().y();
        float inv_dz = 1.0f / r_local.direction().z();

        // Calculate interval parameters for all 3 slabs
        // Use the slab method to determine intersection
        float tx1 = (p_min.x() - r_local.origin().x()) * inv_dx;
        float tx2 = (p_max.x() - r_local.origin().x()) * inv_dx;
        float ty1 = (p_min.y() - r_local.origin().y()) * inv_dy;
        float ty2 = (p_max.y() - r_local.origin().y()) * inv_dy;
        float tz1 = (p_min.z() - r_local.origin().z()) * inv_dz;
        float tz2 = (p_max.z() - r_local.origin().z()) * inv_dz;

        float t_x_min = fminf(tx1, tx2);
        float t_x_max = fmaxf(tx1, tx2);
        float t_y_min = fminf(ty1, ty2);
        float t_y_max = fmaxf(ty1, ty2);
        float t_z_min = fminf(tz1, tz2);
        float t_z_max = fmaxf(tz1, tz2);

        // Find overall entry (t_near) and exit (t_far)
        float t_near = fmaxf(t_x_min, fmaxf(t_y_min, t_z_min));
        float t_far  = fminf(t_x_max, fminf(t_y_max, t_z_max));

        // Miss condition
        if (t_near > t_far) return false;

        // Handle valid range and camera/origin inside the box
        float hit_t = t_near;
        if (hit_t < ray_t.min) {
            hit_t = t_far; // Hit the exit wall from inside
            if (hit_t < ray_t.min || hit_t > ray_t.max) return false;
        } else if (hit_t > ray_t.max) {
            return false;
        }

        // Determine outward normal based on the limiting entry axis
        vec3 outward_normal(0.0f, 0.0f, 0.0f);
        if (hit_t == t_x_min) {
            outward_normal = vec3((inv_dx < 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else if (hit_t == t_y_min) {
            outward_normal = vec3(0.0f, (inv_dy < 0.0f) ? 1.0f : -1.0f, 0.0f);
        } else if (hit_t == t_z_min) {
            outward_normal = vec3(0.0f, 0.0f, (inv_dz < 0.0f) ? 1.0f : -1.0f);
        } else {
            // Exiting from inside (hit_t == t_far)
            if (hit_t == t_x_max) outward_normal = vec3((inv_dx < 0.0f) ? -1.0f : 1.0f, 0.0f, 0.0f);
            else if (hit_t == t_y_max) outward_normal = vec3(0.0f, (inv_dy < 0.0f) ? -1.0f : 1.0f, 0.0f);
            else outward_normal = vec3(0.0f, 0.0f, (inv_dz < 0.0f) ? -1.0f : 1.0f);
        }

        point3 p_local = r_local.at(hit_t);
        // Normalize coordinates to [0, 1] relative to the box dimensions
        float nx = (p_local.x() - p_min.x()) / (p_max.x() - p_min.x());
        float ny = (p_local.y() - p_min.y()) / (p_max.y() - p_min.y());
        float nz = (p_local.z() - p_min.z()) / (p_max.z() - p_min.z());      

        nx = interval(0,1).clamp(nx);
        ny = interval(0,1).clamp(ny);
        nz = interval(0,1).clamp(nz);

        if (outward_normal.x() != 0.0f) {
            // Left (-X) or Right (+X) face
            rec.u = (outward_normal.x() > 0.0f) ? (1.0f - nz) : nz;
            rec.v = ny;
        } else if (outward_normal.y() != 0.0f) {
            // Bottom (-Y) or Top (+Y) face
            rec.u = nx;
            rec.v = (outward_normal.y() > 0.0f) ? (1.0f - nz) : nz;
        } else {
            // Back (-Z) or Front (+Z) face
            rec.u = (outward_normal.z() > 0.0f) ? nx : (1.0f - nx);
            rec.v = ny;
        }
        // Record
        rec.t = hit_t;
        rec.p = r.at(rec.t);
        rec.set_face_normal(r, rotation * outward_normal);
        rec.mat = mat;

        return true;
    }
    __device__ __host__ aabb bounding_box() const override {
        return bbox;
    }
};  

#endif