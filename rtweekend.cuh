#ifndef RTWEEKEND_CUH
#define RTWEEKEND_CUH


#include <cmath>
#include <cfloat>


// CUDA execution space macro
#if defined(__CUDACC__)
    #define HD __host__ __device__
#else
    #define HD
#endif


// Constants (Single-precision float)
#define INFINITY_F (FLT_MAX)
#define PI_F (3.14159265358979323846f)


// Utility Functions
HD inline float degrees_to_radians(float degrees) {
    return degrees * PI_F / 180.0f;
}


__device__ inline unsigned int pcg_hash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Returns a random float between [0, 1) using PCG hash
__device__ inline float rand_float(unsigned int* rng_state) {
    *rng_state = pcg_hash(*rng_state);

    return static_cast<float>(*rng_state >> 8)
         * 5.960464477539063e-8f;
}

__device__ inline float rand_float(float min, float max, unsigned int *rng_state) {
    return min + (max - min) * rand_float(rng_state);
}


// Common Headers
#include "vec3.cuh"
#include "ray.h"
#include "colour.h"
#include "interval.h"

struct light_sample {
    vec3 position;
    vec3 normal;
    colour Le;
    float pdf_area;
};


#endif


