#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.cuh"
#include "bvh.cuh"

class camera {
    public:
        float aspect_ratio = 1.0f;  // Ratio of image width over height
        int   image_width  = 100;   // Rendered image width in pixel count
        int samples_per_pixel = 10;
        int max_depth = 5;
        colour background;

        float vfov = 90.0f;
        point3 look_from = point3(0, 0, 0);
        point3 look_at = point3(0, 0, -1);
        vec3 vup = vec3(0, 1, 0);

        int    image_height;        // Rendered image height
        float pixel_samples_scale;
        point3 center;              // Camera center
        point3 pixel00_loc;         // Location of pixel 0, 0
        vec3   pixel_delta_u;       // Offset to pixel to the right
        vec3   pixel_delta_v;       // Offset to pixel below
        vec3 u, v, w;               // Basis vectors for the camera

        __host__ __device__ void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0f / samples_per_pixel;

            center = look_from;
            float focal_length = (look_at - look_from).length();
            
            // Determine viewport dimensions
            float theta = degrees_to_radians(vfov);
            float h = tanf(theta / 2.0f);
            float viewport_height = 2.0f * h * focal_length;
            float viewport_width = viewport_height * (float(image_width) / image_height);

            w = unit_vector(look_from - look_at);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate horizontal and vertical viewport vectors
            vec3 viewport_u = viewport_width * u;
            vec3 viewport_v = -viewport_height * v;

            // Pixel-to-pixel step vectors
            pixel_delta_u = viewport_u / float(image_width);
            pixel_delta_v = viewport_v / float(image_height);

            // Upper-left pixel location
            vec3 viewport_upper_left = center - (focal_length * w) - viewport_u / 2.0f - viewport_v / 2.0f;
            pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
        }

        __device__ ray get_ray(int i, int j, unsigned int* local_rand_state) const {
            vec3 offset = sample_square(local_rand_state);
            vec3 pixel_sample = pixel00_loc + ((float(i) + offset.x()) * pixel_delta_u) + ((float(j) + offset.y()) * pixel_delta_v);

            vec3 ray_origin = center;
            vec3 ray_direction = pixel_sample - ray_origin;
            return ray(ray_origin, ray_direction);
        }

        __device__ vec3 sample_square(unsigned int* local_rand_state) const {
            return vec3(rand_float(local_rand_state) - 0.5f, rand_float(local_rand_state) - 0.5f, 0.0f);
        }
};

__device__ inline float power_heuristic(float p_f, float p_g) {
    float f2 = p_f * p_f;
    float g2 = p_g * p_g;
    float denom = f2 + g2;
    return (denom > 0.0f) ? (f2 / denom) : 0.0f;
}

__device__ static colour estimate_direct_light(
    const hit_record& rec, 
    const vec3& wo, 
    const flat_bvh* world, 
    const gpu_primitive* d_lights, 
    int num_lights, 
    unsigned int* local_rand_state
) {
    constexpr float epsilon = 0.001f;
    if (num_lights <= 0 || !rec.mat->supports_nee()) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    // Pick a light uniformly from the flat light array
    int light_index = static_cast<int>(rand_float(local_rand_state) * num_lights);
    if (light_index >= num_lights) light_index = num_lights - 1;

    const gpu_primitive& light = d_lights[light_index];
    float pdf_select = 1.0f / static_cast<float>(num_lights);

    // Statically dispatched direct light sampling
    light_sample sample = light.sample_light_point(local_rand_state);

    if (sample.pdf_area <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    vec3 to_light = sample.position - rec.p;
    float distance_squared = to_light.length_squared();
    float distance = sqrtf(distance_squared);
    if (distance_squared <= epsilon * epsilon) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    vec3 wi = to_light / distance;

    float cos_surface = dot(rec.normal, wi);
    if (cos_surface <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    float cos_light = dot(sample.normal, -wi);
    if (cos_light <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    // Compute the unoccluded contribution before BVH traversal.
    float pdf_total = pdf_select * sample.pdf_area;

    if (pdf_total <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }
    colour f = rec.mat->eval(wo, wi, rec);

    float p_light = pdf_total * (distance_squared / cos_light);
    float p_bsdf = rec.mat->scattering_pdf(wo, wi, rec);
    float light_weight = power_heuristic(p_light, p_bsdf);

    colour contribution =
        light_weight * f * sample.Le * (cos_surface * cos_light)
        / (distance_squared * pdf_total);

    // Visibility is the final operation.
    ray shadow_ray(rec.p, wi);

    if (world->occluded(
            shadow_ray,
            interval(epsilon, distance - epsilon))) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    return contribution;

}

__device__ static float compute_light_pdf(
    const hit_record& rec,
    const point3& prev_origin,
    const gpu_primitive* d_lights,
    int num_lights
) {
    if (num_lights <= 0) return 0.0f;

    // Verify whether the hit surface is one of the active NEE emitters
    const gpu_primitive* hit_light = nullptr;
    for (int i = 0; i < num_lights; ++i) {
        if (d_lights[i].object == rec.object) {
            hit_light = &d_lights[i];
            break;
        }
    }
    if (!hit_light) return 0.0f;

    vec3 d_vec = rec.p - prev_origin;
    float dist_sq = d_vec.length_squared();
    if (dist_sq < 1e-8f) return 0.0f;

    vec3 dir = unit_vector(d_vec);

    // Call geometric_normal and area_val through the devirtualized primitive
    float cos_theta_light = dot(hit_light->geometric_normal(rec.p), -dir);
    if (cos_theta_light <= 0.0f) return 0.0f;

    float area = hit_light->area_val();
    if (area <= 0.0f) return 0.0f;

    return (dist_sq / (float(num_lights) * area * cos_theta_light));
}

__device__ colour ray_colour(
    const ray& initial_ray,
    const flat_bvh* world,
    int max_depth,
    unsigned int* local_rand_state,
    const colour& background,
    const gpu_primitive* d_lights,
    int num_lights
) {
    ray cur_ray = initial_ray;
    colour cur_attenuation(1.0f, 1.0f, 1.0f);
    colour radiance(0.0f, 0.0f, 0.0f);

    constexpr int RUSSIAN_ROULETTE_START_DEPTH = 5;
    constexpr float RAY_EPSILON = 0.001f;

    bool previous_used_nee = false;
    float prev_bsdf_pdf = 1.0f;
    point3 prev_origin = initial_ray.origin();

    for (int depth = 0; depth < max_depth; ++depth) {
        hit_record rec;

        // Closest-hit ray query
        if (!world->hit(cur_ray, interval(RAY_EPSILON, INFINITY_F), rec)) {
            radiance += cur_attenuation * background;
            break;
        }

        vec3 wo = -unit_vector(cur_ray.direction());

        // Direct light sampling via NEE
        bool use_nee = rec.mat->supports_nee() && (num_lights > 0);
        if (use_nee) {
            colour direct_light = estimate_direct_light(
                rec,
                wo,
                world,
                d_lights,
                num_lights,
                local_rand_state
            );
            radiance += cur_attenuation * direct_light;
        } 

        // Material scatter bounce
        ray scattered;
        colour attenuation;

        if (!rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
            float mis_weight = 1.0f;
            if (previous_used_nee) {
                float p_light = compute_light_pdf(rec, prev_origin, d_lights, num_lights);
                mis_weight = power_heuristic(prev_bsdf_pdf, p_light);
            }

            radiance += cur_attenuation * rec.mat->emitted() * mis_weight;
            break;
        }

        cur_attenuation = cur_attenuation * attenuation;
        cur_ray = scattered;

        previous_used_nee = use_nee;
        prev_origin = rec.p;
        prev_bsdf_pdf = rec.mat->scattering_pdf(wo, scattered.direction(), rec);

        // Path termination via Russian roulette
        if (depth >= RUSSIAN_ROULETTE_START_DEPTH) {
            float survive_probability = fmaxf(
                cur_attenuation.x(),
                fmaxf(cur_attenuation.y(), cur_attenuation.z())
            );

            survive_probability = fminf(fmaxf(survive_probability, 0.05f), 0.95f);

            if (rand_float(local_rand_state) > survive_probability) {
                break;
            }

            cur_attenuation = cur_attenuation / survive_probability;
        }
    }

    return radiance;
}

#endif