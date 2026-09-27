#ifndef VEC3_CUH
#define VEC3_CUH

class vec3 {
  public:
    float e[3];

    HD vec3() : e{0,0,0} {}
    HD vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    HD float x() const { return e[0]; }
    HD float y() const { return e[1]; }
    HD float z() const { return e[2]; }

    HD vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    HD float operator[](int i) const { return e[i]; }
    HD float& operator[](int i) { return e[i]; }

    HD vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    HD vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    HD vec3& operator/=(float t) {
        return *this *= 1.0f /t;
    }

    HD float length() const {
        return sqrtf(length_squared());
    }

    HD float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    HD bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        constexpr float s = 1e-8f;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }

    __device__ static vec3 random(unsigned int *local_rand_state) {
        return vec3(rand_float(local_rand_state), rand_float(local_rand_state), rand_float(local_rand_state));
    }

    __device__ static vec3 random(float min, float max, unsigned int *local_rand_state) {
        return vec3(rand_float(min, max, local_rand_state), rand_float(min, max, local_rand_state), rand_float(min, max, local_rand_state));
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
/*
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
    */

HD inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

HD inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

HD inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

HD inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

HD inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

HD inline vec3 operator/(const vec3& v, float t) {
    return (1.0f / t) * v;
}

HD inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

HD inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

HD inline vec3 unit_vector(const vec3& v) {
    const float len_sq = v.length_squared();
    return v * rsqrtf(len_sq);
}


// Find algorithm in Notes
__device__ inline vec3 random_unit_vector(unsigned int *local_rand_state) {
    // Generate two uniform random numbers in [0, 1)
    float u1 = rand_float(local_rand_state);
    float u2 = rand_float(local_rand_state);

    // Uniform height distribution z in [-1, 1]
    float z = 1.0f - 2.0f * u2;

    // Radius of cross-section at height z (clamp to 0 to prevent NaN precision errors)
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));

    // Uniform azimuthal angle phi in [0, 2*pi)
    float phi = 2.0f * PI_F * u1;

    float sin_phi, cos_phi;
    __sincosf(phi, &sin_phi, &cos_phi);

    return vec3(r * cos_phi, r * sin_phi, z);
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, unsigned int *local_rand_state) {
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else 
        return -on_unit_sphere;
}

// Find algorithm in Notes
HD inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

// Find algorithm in Notes
HD inline vec3 refract(
    const vec3& uv,
    const vec3& n,
    float eta_i_over_eta_t
) {
    // uv and n must both be normalized.
    // n must oppose the incident direction uv.
    const float cos_theta = fminf(dot(-uv, n), 1.0f);

    // Component tangent to the interface.
    const vec3 r_out_perp =
        eta_i_over_eta_t * (uv + cos_theta * n);

    // Component perpendicular to the interface.
    // The minus sign sends the ray through the boundary.
    const float parallel_length_sq =
        fmaxf(0.0f, 1.0f - r_out_perp.length_squared());

    const vec3 r_out_parallel =
        -sqrtf(parallel_length_sq) * n;

    return r_out_perp + r_out_parallel;
}
#endif