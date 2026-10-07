#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "hittable.h"
#include "texture.h"
#include "device_math.cuh"


class material {
    public:
        __device__ virtual ~material() = default;

        __device__ virtual colour emitted() const {
            return colour(0, 0, 0);
        }
        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state
        ) const {
            return false;
        }
        __device__ virtual bool supports_nee() const {
            return false;
        }

        __device__ virtual colour eval(
            const vec3& wo,
            const vec3& wi,
            const hit_record& rec
        ) const {
            return colour(0.0f, 0.0f, 0.0f);
        }

        __device__ virtual float scattering_pdf(
            const vec3& wo,
            const vec3& wi,
            const hit_record& rec
        ) const {
            return 0.0f;
        }
};

class lambertian : public material {
    public:
        
        __device__ lambertian(texture* tex) : tex(tex) {}

        // Find algorithm in Notes
        __device__ bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state)
        const override {
            auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);
            // catch degenerate scatter direction
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            scattered = ray(rec.p, scatter_direction);
            attenuation = tex->value(rec.u, rec.v, rec.p);
            return true;
        }
        __device__ bool supports_nee() const override {
            return true;
        }

        __device__ colour eval(
            const vec3& wo, // incident ray to the point in rec
            const vec3& wi, // ray going from point to light source
            const hit_record& rec
        ) const override {
            return tex->value(rec.u, rec.v, rec.p) / PI_F;
        }

        __device__ float scattering_pdf(
            const vec3& wo,
            const vec3& wi,
            const hit_record& rec
        ) const override {
            float cosine = dot(rec.normal, unit_vector(wi));
            return fmaxf(0.0f, cosine) / PI_F;
        }
    private:
        texture* tex;
};

// Mirror -> Colour = (1, 1, 1) Fuzz = 0
class metal : public material {
    public:
        __device__ metal(const colour& albedo, float fuzz) : albedo(albedo), fuzz(fmaxf(fminf(fuzz, 1.0f), 0.0f)) {}
        
        // Find algorithm in Notes
        __device__ bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state)
        const override {
            vec3 reflected = reflect(r_in.direction(), rec.normal);

            if (fuzz == 0.0f) {
                scattered = ray(rec.p, reflected);
                attenuation = albedo;
                return dot(reflected, rec.normal) > 0.0f;
            }
            reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_rand_state));
            scattered = ray(rec.p, reflected);
            attenuation = albedo;
            return dot(scattered.direction(), rec.normal) > 0.0f; // checks to make sure that the scattered ray is above the surface, otherwise absorb the ray
        }
    private:
        colour albedo;
        float fuzz;
};

/*
Material -> Refraction index
Air -> 1.0
Water -> 1.31 - 1.33
Glass -> 1.50 - 11.52
Flint Glass -> 1.70
Diamond -> 2.42
*/
class dielectric : public material {
public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    // Find algorithm in Notes
    __device__ bool scatter(
        const ray& r_in,
        const hit_record& rec,
        colour& attenuation,
        ray& scattered,
        unsigned int *local_rand_state
    ) const override {
        attenuation = colour(1.0f, 1.0f, 1.0f);
        vec3 unit_direction = unit_vector(r_in.direction());


        //  If refractive index is 1.0, there is no boundary: transmit straight through
        if (fabsf(refraction_index - 1.0f) < 1e-4f) {
            scattered = ray(rec.p, unit_direction);
            return true;
        }


        float ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;


        // Clamp cosine strictly between 0.0 and 1.0
        float cos_theta = fminf(fmaxf(dot(-unit_direction, rec.normal), 0.0f), 1.0f);
        float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));


        bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;


        // Evaluate reflectance with safe inputs and verified RNG
        float rng_val = rand_float(local_rand_state);
        if (cannot_refract || reflectance(cos_theta, ri) > rng_val) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, ri);
        }


        scattered = ray(rec.p, direction);
        return true;
    }


private:
    float refraction_index;

    __device__ static float reflectance(float cosine, float refraction_index) {
        float r0 = (1.0f - refraction_index)
                 / (1.0f + refraction_index);

        r0 *= r0;

        const float one_minus_cosine = 1.0f - cosine;
        const float one_minus_cosine_2 =
            one_minus_cosine * one_minus_cosine;

        const float one_minus_cosine_5 =
            one_minus_cosine_2
            * one_minus_cosine_2
            * one_minus_cosine;

        return r0 + (1.0f - r0) * one_minus_cosine_5;
    }
};
class diffuse_light : public material {
    public:
        __device__ diffuse_light(colour emit) : emit(emit) {}

        __device__ colour emitted() const override {
            return emit;
        }
    private:
        colour emit;
};

#endif