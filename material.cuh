#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "hittable.h"
#include "texture.h"
#include "device_math.cuh"



struct LambertianData {
    texture *tex;
};

struct MetalData {
    colour albedo;
    float fuzz;
};

struct DielectricData {
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

struct DiffuseLightData {
    colour emit;
};

enum MaterialType : uint8_t {
    Lambertian,
    Metal,
    Dielectric,
    DiffuseLight
};

__device__ __forceinline__ bool scatter_lambertian(const LambertianData& data, const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state) {
    auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);
    // catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;
    scattered = ray(rec.p, scatter_direction);
    attenuation = data.tex->value(rec.u, rec.v, rec.p);
    return true;
}

__device__ __forceinline__ bool scatter_metal(const MetalData& data, const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state) {
    vec3 reflected = reflect(r_in.direction(), rec.normal);

    if (data.fuzz == 0.0f) {
        scattered = ray(rec.p, reflected);
        attenuation = data.albedo;
        return dot(reflected, rec.normal) > 0.0f;
    }
    reflected = unit_vector(reflected) + (data.fuzz * random_unit_vector(local_rand_state));
    scattered = ray(rec.p, reflected);
    attenuation = data.albedo;
    return dot(scattered.direction(), rec.normal) > 0.0f; // checks to make sure that the scattered ray is above the surface, otherwise absorb the ray
}

__device__ __forceinline__ bool scatter_dielectric(const DielectricData& data, const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state) {
    attenuation = colour(1.0f, 1.0f, 1.0f);
    vec3 unit_direction = unit_vector(r_in.direction());


    //  If refractive index is 1.0, there is no boundary: transmit straight through
    if (fabsf(data.refraction_index - 1.0f) < 1e-4f) {
        scattered = ray(rec.p, unit_direction);
        return true;
    }


    float ri = rec.front_face ? (1.0 / data.refraction_index) : data.refraction_index;


    // Clamp cosine strictly between 0.0 and 1.0
    float cos_theta = fminf(fmaxf(dot(-unit_direction, rec.normal), 0.0f), 1.0f);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));


    bool cannot_refract = ri * sin_theta > 1.0f;
    vec3 direction;


    // Evaluate reflectance with safe inputs and verified RNG
    float rng_val = rand_float(local_rand_state);
    if (cannot_refract || data.reflectance(cos_theta, ri) > rng_val) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, ri);
    }


    scattered = ray(rec.p, direction);
    return true;
}


__device__ __forceinline__ float scattering_pdf_lambertian(const vec3& wo, const vec3& wi, const hit_record& rec) {
    float cosine = dot(rec.normal, unit_vector(wi));
    return fmaxf(0.0f, cosine) / PI_F;
}

struct gpu_material {
    MaterialType type;

    union MaterialData {
        LambertianData   lambertian;
        MetalData        metal;
        DielectricData   dielectric;
        DiffuseLightData diffuse_light;

        __host__ __device__
        explicit MaterialData(const LambertianData& value)
            : lambertian{value} {}

        __host__ __device__
        explicit MaterialData(const MetalData& value)
            : metal{
                value.albedo,
                fmaxf(fminf(value.fuzz, 1.0f), 0.0f)
            } {}

        __host__ __device__
        explicit MaterialData(const DielectricData& value)
            : dielectric{value} {}

        __host__ __device__
        explicit MaterialData(const DiffuseLightData& value)
            : diffuse_light{value} {}

        __host__ __device__
        ~MaterialData() = default;
    } data;

    __host__ __device__
    explicit gpu_material(const LambertianData& value)
        : type{MaterialType::Lambertian}, data{value} {}

    __host__ __device__
    explicit gpu_material(const MetalData& value)
        : type{MaterialType::Metal}, data{value} {}

    __host__ __device__
    explicit gpu_material(const DielectricData& value)
        : type{MaterialType::Dielectric}, data{value} {}

    __host__ __device__
    explicit gpu_material(const DiffuseLightData& value)
        : type{MaterialType::DiffuseLight}, data{value} {}

    __host__ __device__
    ~gpu_material() = default;

    __device__ __forceinline__ colour emitted() {

        switch(type) {
            case MaterialType::DiffuseLight:
                return data.diffuse_light.emit;
            default:
                return colour(0.0f, 0.0f, 0.0f);
        }
    }

    __device__ __forceinline__ bool supports_nee() {
        switch(type) {
            case MaterialType::Lambertian:
                return true;
            default:
                return false;
        }
    }

    __device__ __forceinline__ colour eval(const vec3& wo, const vec3& wi, const hit_record& rec) {
        switch(type) {
            case MaterialType::Lambertian:
                return data.lambertian.tex->value(rec.u, rec.v, rec.p) / PI_F;
            default:
                return colour(0.0f, 0.0f, 0.0f);
        }
    }

    __device__ __forceinline__ bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, unsigned int *local_rand_state) {
        switch (type) {
            case MaterialType::Lambertian:
                return scatter_lambertian(data.lambertian, r_in, rec, attenuation, scattered, local_rand_state);
            case MaterialType::Metal:
                return scatter_metal(data.metal, r_in, rec, attenuation, scattered, local_rand_state);
            case MaterialType::Dielectric:
                return scatter_dielectric(data.dielectric, r_in, rec, attenuation, scattered, local_rand_state);
            default:
                return false;
        }
    }

    __device__ __forceinline__ float scattering_pdf(const vec3& wo, const vec3& wi, const hit_record& rec) {
        switch (type) {
            case MaterialType::Lambertian:
                return scattering_pdf_lambertian(wo, wi, rec);
            default:
                return 0.0f;
        }
    }

};

#endif