#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.cuh"
#include "bvh.cuh"


class camera {
    public:
        float aspect_ratio = 1.0f;  // Ratio of image width over height
        int    image_width  = 100;  // Rendered image width in pixel count
        int samples_per_pixel = 10;
        int max_depth = 5;
        colour background;

        float vfov = 90.0f;
        point3 look_from = point3(0, 0, 0);
        point3 look_at = point3(0, 0, -1);
        vec3 vup = vec3(0, 1, 0);


        int    image_height;   // Rendered image height
        float pixel_samples_scale;
        point3 center;         // Camera center
        point3 pixel00_loc;    // Location of pixel 0, 0
        vec3   pixel_delta_u;  // Offset to pixel to the right
        vec3   pixel_delta_v;  // Offset to pixel below
        vec3 u, v, w; // Basis vectors for the camera

        __host__ __device__ void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = look_from;
            float focal_length = (look_at - look_from).length();
            // Determine viewport dimensions.
            float theta = degrees_to_radians(vfov);
            float h = tanf(theta/2);
            float viewport_height = 2 * h * focal_length;
            float viewport_width = viewport_height * (float(image_width)/image_height);

            w = unit_vector(look_from - look_at);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            vec3 viewport_u = viewport_width * u;
            vec3 viewport_v = -viewport_height * v;

            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Calculate the camera lense dimensions

            // Calculate the location of the upper left pixel.
            vec3 viewport_upper_left =
                center - (focal_length * w) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        }

        __device__ ray get_ray(int i, int j, unsigned int* local_rand_state) const {
            vec3 offset = sample_square(local_rand_state); // gets a random offset for anti-aliasing
            vec3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

            vec3 ray_origin = center;
            vec3 ray_direction = pixel_sample - ray_origin;
            return ray(ray_origin, ray_direction);
        }

        __device__ vec3 sample_square(unsigned int* local_rand_state) const {
            // Returns a vector to a random point in the [-0.5, -0.5] -> [0.5, 0.5] unit square.
            return vec3(rand_float(local_rand_state) - 0.5f, rand_float(local_rand_state) - 0.5f, 0.0f);
        }
        


};

__device__ inline float power_heuristic(float p_f, float p_g) {
    float f2 = p_f * p_f;
    float g2 = p_g * p_g;
    float denom = f2 + g2;
    return (denom > 0.0f) ? (f2 / denom) : 0.0f;
}

__device__ colour estimate_direct_light(const hit_record& rec, const vec3& wo, const flat_bvh* world, quad* const* d_lights, int num_lights, unsigned int* local_rand_state) {
    constexpr float epsilon = 0.001f;
    if (num_lights <= 0 || !rec.mat->supports_nee()) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    // Pick a random light from the list of lights
    int light_index = static_cast<int>(rand_float(local_rand_state) * num_lights);
    const quad* light = d_lights[light_index];
    // Probability of selecting that specific light
    float pdf_select = 1.0f / static_cast<float>(num_lights);

    // Sample a random point on the light
    light_sample sample = light->sample_light_point(local_rand_state);

    if (sample.pdf_area <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    vec3 to_light = sample.position - rec.p;
    float distance_squared = to_light.length_squared();
    float distance = sqrtf(distance_squared);
    if (distance_squared <= epsilon * epsilon) {
        // return black to avoid divide by 0 error
        return colour(0.0f, 0.0f, 0.0f);
    }
    // Normalissed vector pointing from the hitpoint to light sample
    vec3 wi = to_light / distance;

    // How much of the receiving surface faces the light
    float cos_surface = dot(rec.normal, wi);

    if (cos_surface <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    // How much of the light's emitting face points towards the receiving surface
    float cos_light = dot(sample.normal, -wi);

    if (cos_light <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    // Shadow ray emitted from point of collision to the sample point
    ray shadow_ray(rec.p, wi);
    bool blocked = world->occluded(shadow_ray, interval(0.001f, distance - 0.001f));

    if (blocked) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    // Determine how the receiving material scatters the received in the direction -wi
    // Is sent back to the camera in the direction w0
    colour f = rec.mat->eval(wo, wi, rec);

    // Probability of picking the specific point on that specific light
    float pdf_total = pdf_select * sample.pdf_area;

    if (pdf_total <= 0.0f) {
        return colour(0.0f, 0.0f, 0.0f);
    }

    float p_light = pdf_total * (distance_squared / cos_light);
    float p_bsdf = rec.mat->scattering_pdf(wo, wi, rec);
    float light_weight = power_heuristic(p_light, p_bsdf);

    // Uniform area-light using Monte Carlo estimation
    return light_weight * f * sample.Le * (cos_surface * cos_light) / (distance_squared * pdf_total);
}



__device__ float compute_light_pdf(
    const hit_record& rec,
    const point3& prev_origin,
    quad* const* d_lights,
    int num_lights
) {
    if (num_lights <= 0) return 0.0f;

    // 1. Verify the hit object is actually a registered NEE light
    // (Requires rec.object to be set in hit_record during world->hit)
    const quad* hit_quad = nullptr;
    for (int i = 0; i < num_lights; ++i) {
        if (d_lights[i] == rec.object) {
            hit_quad = d_lights[i];
            break;
        }
    }
    if (!hit_quad) return 0.0f;

    // 2. Vector pointing from the previous surface vertex to the light hit point
    vec3 d_vec = rec.p - prev_origin;
    float dist_sq = d_vec.length_squared();
    if (dist_sq < 1e-8f) return 0.0f;

    vec3 dir = unit_vector(d_vec);

    // 3. Cosine of angle between incoming ray and quad emitter's normal
    // If the quad is one-sided, hits from behind cannot be sampled by NEE
    float cos_theta_light = dot(hit_quad->geometric_normal(), -dir);
    if (cos_theta_light <= 0.0f) return 0.0f;

    // 4. Area of the quad
    float area = hit_quad->area_val();
    if (area <= 0.0f) return 0.0f;

    // 5. Convert area PDF to solid-angle PDF:
    // p_area = 1.0f / (num_lights * area)
    // p_omega = p_area * (dist_sq / cos_theta_light)
    return (dist_sq / (num_lights * area * cos_theta_light));
}

__device__ colour ray_colour(
    const ray& initial_ray,
    const flat_bvh* world,
    int max_depth,
    unsigned int* local_rand_state,
    const colour& background,
    quad* const* d_lights,
    int num_lights
) {
    ray cur_ray = initial_ray;

    // Product of all ordinary camera-path bounce weights so far.
    colour cur_attenuation(1.0f, 1.0f, 1.0f);

    // Sum of all radiance contributions discovered along this path.
    colour radiance(0.0f, 0.0f, 0.0f);

    constexpr int RUSSIAN_ROULETTE_START_DEPTH = 5;
    constexpr float RAY_EPSILON = 0.001f;

    // This tracks whether the previous vertex used NEE light sampling.
    // It is needed for the temporary no-MIS emission rule.
    bool previous_used_nee = false;
    // Stored from the scattering bounce:
    float prev_bsdf_pdf = 1;      // p_bsdf evaluated at the surface
    point3 prev_origin = initial_ray.origin();       // x_prev

    for (int depth = 0; depth < max_depth; ++depth) {
        hit_record rec;

        // -----------------------------------------------------------------
        // 1. Ordinary closest-hit ray query
        // -----------------------------------------------------------------
        if (!world->hit(
                cur_ray,
                interval(RAY_EPSILON, INFINITY_F),
                rec
            )) {
            // The current path escaped. Add environmental/background light.
            radiance += cur_attenuation * background;
            break;
        }

        // wo points from the current hit point back toward the camera /
        // previous path vertex.
        vec3 wo = -unit_vector(cur_ray.direction());

        // -----------------------------------------------------------------
        // 2. NEE: explicitly sample one light connection at this vertex
        // -----------------------------------------------------------------
        bool use_nee =
            rec.mat->supports_nee() &&
            num_lights > 0;
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
        // -----------------------------------------------------------------
        // 3. Ordinary BSDF path continuation
        // -----------------------------------------------------------------
        ray scattered;
        colour attenuation;

        if (!rec.mat->scatter(
                cur_ray,
                rec,
                attenuation,
                scattered,
                local_rand_state
            )) {
            
            float mis_weight = 1.0f;
            if (previous_used_nee) {
                // Compute the chance that the previous shadow ray hit this quad

                float p_light = compute_light_pdf(rec, prev_origin, d_lights, num_lights);
                mis_weight = power_heuristic(prev_bsdf_pdf, p_light);
            }

            radiance += cur_attenuation * rec.mat->emitted() * mis_weight;
            break;
        }

        // -----------------------------------------------------------------
        // 4. Update ordinary-path throughput and continue
        // -----------------------------------------------------------------
        cur_attenuation = cur_attenuation * attenuation;
        cur_ray = scattered;

        // The next ray originated at a material that used NEE iff this
        // current material was a Lambertian NEE-capable material.
        previous_used_nee = use_nee;
        prev_origin = rec.p;
        prev_bsdf_pdf = rec.mat->scattering_pdf(wo, scattered.direction(), rec);

        // -----------------------------------------------------------------
        // 5. Russian roulette: only terminate future path continuation
        // -----------------------------------------------------------------
        if (depth >= RUSSIAN_ROULETTE_START_DEPTH) {
            float survive_probability = fmaxf(
                cur_attenuation.x(),
                fmaxf(
                    cur_attenuation.y(),
                    cur_attenuation.z()
                )
            );

            survive_probability = fminf(
                fmaxf(survive_probability, 0.05f),
                0.95f
            );

            if (rand_float(local_rand_state) > survive_probability) {
                // Do NOT return black here.
                //
                // You may already have added direct-light contributions
                // from earlier vertices to `radiance`.
                break;
            }

            cur_attenuation =
                cur_attenuation / survive_probability;
        }
    }

    return radiance;
}


#endif