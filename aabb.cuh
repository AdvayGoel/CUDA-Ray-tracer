#ifndef AABB_CUH
#define AABB_CUH
#include "device_math.cuh"

class aabb {
    public:
        interval x, y, z; 

        HD    aabb() {}

        HD aabb(const interval& x, const interval& y, const interval& z)
        : x(x), y(y), z(z) {
            pad_to_minimums(); // pad to ensure no infinitely small dimension
        }

        HD aabb(const point3& a, const point3& b) {
            x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
            y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
            z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
            pad_to_minimums();
        }

        HD aabb(const aabb& box0, const aabb& box1) {
            x = interval(box0.x, box1.x);
            y = interval(box0.y, box1.y);
            z = interval(box0.z, box1.z);
            pad_to_minimums();
        }

        HD const interval& axis_interval(int n) const {
            if (n == 1) return y;
            if (n == 2) return z;
            else return x;
        }

        // Find algorithm in Notes
        HD bool hit(const ray& r, interval ray_t, float inv_dir_x, float inv_dir_y, float inv_dir_z) const {
            // takes in reciprocal of the direction to prevent repeated floating point division
            // uses slab method for determining intersection
            float t0_x = (x.min - r.origin().x()) * inv_dir_x;
            float t1_x = (x.max - r.origin().x()) * inv_dir_x;
            float tmin = fminf(t0_x, t1_x);
            float tmax = fmaxf(t0_x, t1_x);

            float t0_y = (y.min - r.origin().y()) * inv_dir_y;
            float t1_y = (y.max - r.origin().y()) * inv_dir_y;
            tmin = fmaxf(tmin, fminf(t0_y, t1_y));
            tmax = fminf(tmax, fmaxf(t0_y, t1_y));

            float t0_z = (z.min - r.origin().z()) * inv_dir_z;
            float t1_z = (z.max - r.origin().z()) * inv_dir_z;
            tmin = fmaxf(tmin, fminf(t0_z, t1_z));
            tmax = fminf(tmax, fmaxf(t0_z, t1_z));

            return (tmax >= fmaxf(tmin, ray_t.min)) && (tmin < ray_t.max);     
        }

    private:
        HD void pad_to_minimums() {
            constexpr float delta = 0.0001;
            if (x.size() < delta) x = x.expand(delta);
            if (y.size() < delta) y = y.expand(delta);
            if (z.size() < delta) z = z.expand(delta);
        }


};

#endif