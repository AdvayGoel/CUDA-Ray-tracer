#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h" 

class 
sphere : public hittable {
  public:
    point3 center;
    float radius;
    material* mat;
    aabb bbox;

    __device__ sphere(const point3& center, float radius, material* mat) : center(center), radius(fmaxf(1e-6f,radius)), mat(mat) {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }

    // Find algorithm in Notes
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0.0f)
            return false;

        auto sqrtd = sqrtf(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;
        return true;
    }
    __device__ __host__ aabb bounding_box() const override {
        return bbox;
    }

    __device__ light_sample sample_light_point(unsigned int* local_rand_state) const override {
            
        light_sample sample;
        vec3 random_unit = random_unit_vector(local_rand_state);
        sample.position = radius * random_unit;
        sample.pdf_area = 1.0f / area_val();
        sample.normal = geometric_normal(sample.position);
        sample.Le = mat->emitted();
        return sample;
    }

    __device__ vec3 geometric_normal(const point3 at) const override {
        return unit_vector(at - center);
    }

    __device__  float area_val() const override{
        return 4 * PI_F * radius * radius;
    }

    private:
        // Find algorithm in Notes
        __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>  
            float theta = acosf(-p.y());
            float phi = atan2f(-p.z(), p.x()) + PI_F;

            u = phi / (2.0f * PI_F);
            v = theta / PI_F;
        }
};

#endif