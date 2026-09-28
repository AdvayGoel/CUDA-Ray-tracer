#ifndef QUAD_H
#define QUAD_H

#include "hittable.h"

class quad : public hittable {
    public:

        __device__ quad(const point3& Q, const vec3& u, const vec3& v, material* mat) : Q(Q), u(u), v(v), mat(mat) {
            aabb bbox_diagonal_1 = aabb(Q, Q + u + v);
            aabb bbox_diagonal_2 = aabb(Q + u, Q + v);
            bbox = aabb(bbox_diagonal_1, bbox_diagonal_2);

            vec3 n = cross(u, v);
            area = n.length();
            w = n / dot(n, n);
            normal = unit_vector(n);
            D = dot(normal, Q);
        }
        
        HD aabb bounding_box() const override { return bbox; };

        // Find algorithm in Notes
        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            float denom = dot(normal, r.direction());

            // if plane and ray are parallel and hence don't meet
            if (fabsf(denom) < 1e-8)
                return false;
            
            float t = (D - dot(normal, r.origin())) / denom;
            // ray meets plane outside ray interval
            if (!ray_t.contains(t)) 
                return false;

            point3 intersection = r.at(t);
            vec3 planar_hitpt_vector = intersection - Q;
            float alpha = dot(w, cross(planar_hitpt_vector, v));
            float beta = dot(w, cross(u, planar_hitpt_vector));

            interval unit_interval = interval(0, 1);

            if (!unit_interval.contains(alpha) || !unit_interval.contains(beta))
                return false;


            rec.t = t;
            rec.p = intersection;
            rec.u = alpha;
            rec.v = beta;
            rec.mat = mat;
            rec.set_face_normal(r, normal);

            return true;
        }

        __device__ light_sample sample_light_point(unsigned int* local_rand_state) const {
            
            light_sample sample;
            float alpha = rand_float(local_rand_state);
            float beta = rand_float(local_rand_state);

            sample.position = Q + alpha * u + beta * v;
            sample.normal = normal;
            sample.pdf_area = 1.0f / area;
            sample.Le = mat->emitted();
            return sample;
        }

        __device__ vec3 geometric_normal() const {
            return normal;
        }

        __device__  float area_val() const {
            return area;
        }

    private:
        point3 Q; // Starting point
        vec3 u, v; // Vectors for both sides
        vec3 w;
        material* mat;
        aabb bbox;
        vec3 normal; // Normal vector to the plane
        float D; // Constant in equation Ax + By + Cz = D
        float area;

};

#endif