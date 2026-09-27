#ifndef COLOUR_H
#define COLOUR_H

#include "interval.h"
#include "vec3.cuh"

using colour = vec3;

// Prevents images from becoming too bright
HD inline float linear_to_gamma(float linear_component) {
    if (linear_component > 0)
        return sqrtf(linear_component);
    return 0.0f;
}

void write_colour(std::ostream& out, const colour& pixel_colour) {
    auto r = pixel_colour.x();
    auto g = pixel_colour.y();
    auto b = pixel_colour.z();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    static const interval intensity(0.000f, 0.999f);
    int rbyte = int(255.999 * intensity.clamp(r));
    int gbyte = int(255.999 * intensity.clamp(g));
    int bbyte = int(255.999 * intensity.clamp(b));

    out << rbyte << ' '<< gbyte << ' '<< bbyte << '\n';
}

#endif