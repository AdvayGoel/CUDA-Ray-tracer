#ifndef INTERVAL_H
#define INTERVAL_H

#include "device_math.cuh"

#if defined(__CUDACC__)
    #define HD __host__ __device__
#else
    #define HD
#endif

class interval {
    public:
        float min, max;
        
        HD interval() : min(+INFINITY_F), max (-INFINITY_F) {}

        HD interval(float min, float max) : min(min), max(max) {}

        HD interval(const interval& a, const interval& b) {
            min = a.min <= b.min ? a.min : b.min;
            max = a.max >= b.max ? a.max : b.max;
        }

        HD float size() const {
            return max - min;
        }

        HD bool contains(float x) {
            return min <= x && x <= max;
        }

        HD bool surrounds(float x) {
            return min < x && x < max;
        }

        HD float clamp(float x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        HD interval expand(float delta) const {
            float padding = delta / 2.0f;
            return interval(min - padding, max + padding);
        }

        static const interval empty, universe;
};

const interval interval::empty = interval(+INFINITY_F, -INFINITY_F);
const interval interval:: universe = interval(-INFINITY_F, +INFINITY_F);

#endif
