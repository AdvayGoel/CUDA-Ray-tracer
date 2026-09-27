#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.cuh"
#include "hittable.h"
#include "rtweekend.cuh"
#include <vector>

class hittable_list : public hittable {
    public:
        hittable** objects;
        int list_size;

        __device__ hittable_list() : objects(nullptr), list_size(0) {}
        __device__ hittable_list(hittable** objects, int n) : objects(objects), list_size(n) {
            for (int i = 0; i < list_size; i++) {
                bbox = aabb(bbox, objects[i]->bounding_box());
            }
        }

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for (int i = 0; i < list_size; i++) {
                if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }

        __device__ aabb bounding_box() const override { return bbox; }
    private:
        aabb bbox;
};

#endif 
