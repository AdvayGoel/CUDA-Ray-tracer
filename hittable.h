#ifndef HITTABLE_H
#define HITTABLE_H

#include "device_math.cuh"
#include "aabb.cuh"
#include "mat3.cuh"
#include <inttypes.h>

class hittable;
struct gpu_material;


class hit_record {
    public:
        point3 p;
        vec3 normal;
        gpu_material* mat;
        float t;
        float u;
        float v;
        bool front_face;
        const hittable* object = nullptr;

        // sets the normal to the opposite direction to the ray direction.
        HD void set_face_normal(const ray& r, const vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0.0f;
            normal = front_face ? outward_normal : -outward_normal;
        }
};


// De-Virtualised hittable objects

struct SphereData {
    point3 center;
    float radius;
};

struct QuadData {
    vec3 Q;        // Corner origin point
    vec3 u;        // Edge vector 1
    vec3 v;        // Edge vector 2
    vec3 w;        // Precalculated cached vector: normal / dot(normal, cross(u, v))
    vec3 normal;    // Precomputed normal to the quad
    float D;       // Plane equation constant: dot(normal, Q)
    float area;

};

struct CuboidData {
    vec3 p_min;
    vec3 p_max;
    mat3 rotation;
    mat3 inverse_rotation;
    point3 center;
    float area;
};

enum PrimitiveType : uint8_t {
    Sphere,
    Quad,
    Cuboid,
    Transform
};

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

__device__ __forceinline__ bool hit_sphere(const SphereData& sphere, const ray& r, interval ray_t, hit_record& rec) {
        vec3 oc = sphere.center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - sphere.radius * sphere.radius;

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
        vec3 outward_normal = (rec.p - sphere.center) / sphere.radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        return true;
}


__device__ __forceinline__ bool hit_quad(const QuadData& quad, const ray& r, interval ray_t, hit_record& rec) {
    float denom = dot(quad.normal, r.direction());

    // if plane and ray are parallel and hence don't meet
    if (fabsf(denom) < 1e-8)
        return false;
    
    float t = (quad.D - dot(quad.normal, r.origin())) / denom;
    // ray meets plane outside ray interval
    if (!ray_t.contains(t)) 
        return false;

    point3 intersection = r.at(t);
    vec3 planar_hitpt_vector = intersection - quad.Q;
    float alpha = dot(quad.w, cross(planar_hitpt_vector, quad.v));
    float beta = dot(quad.w, cross(quad.u, planar_hitpt_vector));

    interval unit_interval = interval(0, 1);

    if (!unit_interval.contains(alpha) || !unit_interval.contains(beta))
        return false;


    rec.t = t;
    rec.p = intersection;
    rec.u = alpha;
    rec.v = beta;
    rec.set_face_normal(r, quad.normal);

    return true;
}

__device__ __forceinline__ bool hit_cuboid(const CuboidData& cuboid, const ray& r, interval ray_t, hit_record& rec) {
    ray r_local = ray(cuboid.inverse_rotation * (r.origin() - cuboid.center), cuboid.inverse_rotation * r.direction());

    // Pre-calculate the reciprocal of the direction to avoid repeated floating point division
    float inv_dx = 1.0f / r_local.direction().x();
    float inv_dy = 1.0f / r_local.direction().y();
    float inv_dz = 1.0f / r_local.direction().z();

    // Calculate interval parameters for all 3 slabs
    // Use the slab method to determine intersection
    vec3 p_min = cuboid.p_min;
    vec3 p_max = cuboid.p_max;
    float tx1 = (p_min.x() - r_local.origin().x()) * inv_dx;
    float tx2 = (p_max.x() - r_local.origin().x()) * inv_dx;
    float ty1 = (p_min.y() - r_local.origin().y()) * inv_dy;
    float ty2 = (p_max.y() - r_local.origin().y()) * inv_dy;
    float tz1 = (p_min.z() - r_local.origin().z()) * inv_dz;
    float tz2 = (p_max.z() - r_local.origin().z()) * inv_dz;

    float t_x_min = fminf(tx1, tx2);
    float t_x_max = fmaxf(tx1, tx2);
    float t_y_min = fminf(ty1, ty2);
    float t_y_max = fmaxf(ty1, ty2);
    float t_z_min = fminf(tz1, tz2);
    float t_z_max = fmaxf(tz1, tz2);

    // Find overall entry (t_near) and exit (t_far)
    float t_near = fmaxf(t_x_min, fmaxf(t_y_min, t_z_min));
    float t_far  = fminf(t_x_max, fminf(t_y_max, t_z_max));

    // Miss condition
    if (t_near > t_far) return false;

    // Handle valid range and camera/origin inside the box
    float hit_t = t_near;
    if (hit_t < ray_t.min) {
        hit_t = t_far; // Hit the exit wall from inside
        if (hit_t < ray_t.min || hit_t > ray_t.max) return false;
    } else if (hit_t > ray_t.max) {
        return false;
    }

    // Determine outward normal based on the limiting entry axis
    vec3 outward_normal(0.0f, 0.0f, 0.0f);
    if (hit_t == t_x_min) {
        outward_normal = vec3((inv_dx < 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
    } else if (hit_t == t_y_min) {
        outward_normal = vec3(0.0f, (inv_dy < 0.0f) ? 1.0f : -1.0f, 0.0f);
    } else if (hit_t == t_z_min) {
        outward_normal = vec3(0.0f, 0.0f, (inv_dz < 0.0f) ? 1.0f : -1.0f);
    } else {
        // Exiting from inside (hit_t == t_far)
        if (hit_t == t_x_max) outward_normal = vec3((inv_dx < 0.0f) ? -1.0f : 1.0f, 0.0f, 0.0f);
        else if (hit_t == t_y_max) outward_normal = vec3(0.0f, (inv_dy < 0.0f) ? -1.0f : 1.0f, 0.0f);
        else outward_normal = vec3(0.0f, 0.0f, (inv_dz < 0.0f) ? -1.0f : 1.0f);
    }

    point3 p_local = r_local.at(hit_t);
    // Normalize coordinates to [0, 1] relative to the box dimensions
    float nx = (p_local.x() - p_min.x()) / (p_max.x() - p_min.x());
    float ny = (p_local.y() - p_min.y()) / (p_max.y() - p_min.y());
    float nz = (p_local.z() - p_min.z()) / (p_max.z() - p_min.z());      

    nx = interval(0,1).clamp(nx);
    ny = interval(0,1).clamp(ny);
    nz = interval(0,1).clamp(nz);

    if (outward_normal.x() != 0.0f) {
        // Left (-X) or Right (+X) face
        rec.u = (outward_normal.x() > 0.0f) ? (1.0f - nz) : nz;
        rec.v = ny;
    } else if (outward_normal.y() != 0.0f) {
        // Bottom (-Y) or Top (+Y) face
        rec.u = nx;
        rec.v = (outward_normal.y() > 0.0f) ? (1.0f - nz) : nz;
    } else {
        // Back (-Z) or Front (+Z) face
        rec.u = (outward_normal.z() > 0.0f) ? nx : (1.0f - nx);
        rec.v = ny;
    }
    // Record
    rec.t = hit_t;
    rec.p = r.at(rec.t);
    rec.set_face_normal(r, cuboid.rotation * outward_normal);
    return true;   
}


__device__ __forceinline__
light_sample sample_sphere(
    const SphereData& sphere,
    unsigned int* local_rand_state
) {
    light_sample sample;

    vec3 random_unit = random_unit_vector(local_rand_state);

    sample.position =
        sphere.center + sphere.radius * random_unit;

    sample.normal = random_unit;

    sample.pdf_area =
        1.0f / (4.0f * PI_F * sphere.radius * sphere.radius);

    return sample;
}
__device__ __forceinline__ light_sample sample_quad(const QuadData& quad, unsigned int* local_rand_state) {
        light_sample sample;
        float alpha = rand_float(local_rand_state);
        float beta = rand_float(local_rand_state);

        sample.position = quad.Q + alpha * quad.u + beta * quad.v;
        sample.normal = quad.normal;
        sample.pdf_area = 1.0f / quad.area;
        return sample;  
}

__device__ __forceinline__ light_sample sample_cuboid(const CuboidData& quad, unsigned int* local_rand_state) {
            light_sample sample;
            vec3 p_max = quad.p_max;
            vec3 p_min = quad.p_min;
            vec3 size = vec3(p_max.x() - p_min.x(), p_max.y() - p_min.y(), p_max.z() - p_min.z());
            float ax = size.y() * size.z();
            float ay = size.x() * size.z();
            float az = size.x() * size.y();

            sample.pdf_area = 1.0f / quad.area;
            float rand_area_scale = rand_float(local_rand_state);
            float r = rand_area_scale * quad.area;
            float u2 = rand_float(local_rand_state);
            float u3 = rand_float(local_rand_state);
            if (r < ax) { // -X face
                sample.position = point3(p_min.x(), p_min.y() + u2 * size.y(), p_min.z() + u3 * size.z());
                sample.normal = vec3(-1.0f, 0.0f, 0.0f);
            } else if (r < 2.0f * ax) { // +X face
                sample.position = point3(p_max.x(), p_min.y() + u2 * size.y(), p_min.z() + u3 * size.z());
                sample.normal = vec3(1.0f, 0.0f, 0.0f);
            } else if (r < 2.0f * ax + ay) { // -Y face
                sample.position = point3(p_min.x() + u2 * size.x(), p_min.y(), p_min.z() + u3 * size.z());
                sample.normal = vec3(0.0f, -1.0f, 0.0f);
            } else if (r < 2.0f * (ax + ay)) { // +Y face
                sample.position = point3(p_min.x() + u2 * size.x(), p_max.y(), p_min.z() + u3 * size.z());
                sample.normal = vec3(0.0f, 1.0f, 0.0f);
            } else if (r < 2.0f * (ax + ay) + az) { // -Z face
                sample.position = point3(p_min.x() + u2 * size.x(), p_min.y() + u3 * size.y(), p_min.z());
                sample.normal = vec3(0.0f, 0.0f, -1.0f);
            } else { // +Z face
                sample.position = point3(p_min.x() + u2 * size.x(), p_min.y() + u3 * size.y(), p_max.z());
                sample.normal = vec3(0.0f, 0.0f, 1.0f);
            }
            sample.position = quad.rotation * sample.position + quad.center;
            sample.normal = quad.rotation * sample.normal;
            return sample;
}

__device__ __forceinline__ float area_sphere(const SphereData& sphere) {
    return 4 * PI_F * sphere.radius * sphere.radius;
}
__device__ __forceinline__ float area_quad(const QuadData& quad) {
    return quad.area;
}
__device__ __forceinline__ float area_cuboid(const CuboidData& cuboid) {
    return cuboid.area;
}
__device__ __forceinline__ vec3 geometric_normal_sphere(const SphereData& sphere, const point3 at) {
    return unit_vector(at - sphere.center);
}

__device__ __forceinline__ vec3 geometric_normal_quad(const QuadData& quad, const point3 at) {
    return quad.normal;
}

__device__ __forceinline__ vec3 geometric_normal_cuboid(const CuboidData& cuboid, const point3 at) {
            point3 p = cuboid.inverse_rotation * (at - cuboid.center); // find local space point
            vec3 p_max = cuboid.p_max;
            vec3 p_min = cuboid.p_min;
            float d_min_x = fabsf(p.x() - p_min.x());
            float d_max_x = fabsf(p.x() - p_max.x());
            float d_min_y = fabsf(p.y() - p_min.y());
            float d_max_y = fabsf(p.y() - p_max.y());
            float d_min_z = fabsf(p.z() - p_min.z());
            float d_max_z = fabsf(p.z() - p_max.z());

            float min_dist = d_min_x;
            vec3 normal = vec3(-1.0f, 0.0f, 0.0f);

            if (d_max_x < min_dist) { 
                min_dist = d_max_x;
                normal = vec3( 1.0f,  0.0f,  0.0f); 
            }
            if (d_min_y < min_dist) { 
                min_dist = d_min_y; 
                normal = vec3( 0.0f, -1.0f,  0.0f); 
            }
            if (d_max_y < min_dist) { 
                min_dist = d_max_y; 
                normal = vec3( 0.0f,  1.0f,  0.0f); 
            }
            if (d_min_z < min_dist) { 
                min_dist = d_min_z; 
                normal = vec3( 0.0f,  0.0f, -1.0f); 
            }
            if (d_max_z < min_dist) {
                normal = vec3( 0.0f,  0.0f,  1.0f); 
            }
            normal = cuboid.rotation * normal; // transform back to world space
            return normal;
}


struct gpu_primitive {
    PrimitiveType type;
    gpu_material* mat;
    hittable *object;
    colour emission;
    union PrimitiveData {
        SphereData   sphere;
        QuadData     quad;
        CuboidData   cuboid;
        // transform if needed
        __host__ __device__ PrimitiveData() {}
        __host__ __device__ ~PrimitiveData() {}
    } data;


    // Explicit default constructor
    __host__ __device__ gpu_primitive() {}

    // Explicit destructor
    __host__ __device__ ~gpu_primitive() {}

    __device__ __forceinline__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        bool is_hit = false;
        switch (type) {
            case PrimitiveType::Sphere:
                is_hit = hit_sphere(data.sphere, r, ray_t, rec);
                break;
            case PrimitiveType::Quad:
                is_hit = hit_quad(data.quad, r, ray_t, rec);
                break;
            case PrimitiveType::Cuboid:
                is_hit = hit_cuboid(data.cuboid, r, ray_t, rec);
                break;
        }
        if (is_hit) {
            rec.mat = mat; // Attach the existing material pointer
            rec.object = object;
        }
        return is_hit;
    }


    __device__ __forceinline__ light_sample sample_light_point(unsigned int* local_rand_state) const {
        light_sample sample;
        switch (type) {
            case PrimitiveType::Sphere:
                sample = sample_sphere(data.sphere, local_rand_state);
                break;
            case PrimitiveType::Quad:
                sample = sample_quad(data.quad, local_rand_state);
                break;
            case PrimitiveType::Cuboid:
                sample = sample_cuboid(data.cuboid, local_rand_state);
                break;
        }
        sample.Le = emission;
        return sample;
    }

    __device__ __forceinline__
    vec3 geometric_normal(const point3 at) const {
        switch (type) {
            case PrimitiveType::Sphere:
                return geometric_normal_sphere(data.sphere, at);

            case PrimitiveType::Quad:
                return geometric_normal_quad(data.quad, at);

            case PrimitiveType::Cuboid:
                return geometric_normal_cuboid(data.cuboid, at);

            default:
                // Converted render primitives should not remain Transform.
                return vec3(0.0f, 0.0f, 0.0f);
        }
    }

    __device__ __forceinline__ float area_val() const {
        float area;
        switch (type) {
            case PrimitiveType::Sphere:
                area = area_sphere(data.sphere);
                break;
            case PrimitiveType::Quad:
                area = area_quad(data.quad);
                break;
            case PrimitiveType::Cuboid:
                area = area_cuboid(data.cuboid);
                break;
        }
        return area;
    }
};

class hittable {
    public:
        
        __device__ __host__ virtual ~hittable() = default; // destructor

        __device__ __host__ virtual aabb bounding_box() const = 0;

        __device__ virtual PrimitiveType get_type() const = 0;
    };

#endif
