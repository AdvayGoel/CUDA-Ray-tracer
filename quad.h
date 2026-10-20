#ifndef QUAD_H
#define QUAD_H

#include "hittable.h"

class quad : public hittable {
    public:
        QuadData data;
        gpu_material* mat;
        aabb bbox;

        __device__ quad(const point3& Q, const vec3& u, const vec3& v, gpu_material* mat) : mat(mat) {
            // 1. Assign the geometry vectors!
            data.Q = Q;
            data.u = u;
            data.v = v;

            aabb bbox_diagonal_1 = aabb(Q, Q + u + v);
            aabb bbox_diagonal_2 = aabb(Q + u, Q + v);
            bbox = aabb(bbox_diagonal_1, bbox_diagonal_2);

            vec3 n = cross(u, v);
            data.area = n.length();
            data.w = n / dot(n, n);
            data.normal = unit_vector(n);
            data.D = dot(data.normal, Q);
        }

        HD aabb bounding_box() const override { return bbox; };

        __device__ PrimitiveType get_type() const override { return PrimitiveType::Quad; }    
    
};

#endif