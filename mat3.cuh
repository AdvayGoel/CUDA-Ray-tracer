#ifndef MAT3_CUH
#define MAT3_CUH

#include "vec3.cuh"

class mat3 {
private:
    float m[9]; // Row-major: elements [0..2], [3..5], [6..8]

public:
    // Default constructor (identity matrix)
    __host__ __device__ mat3() {
        m[0] = 1.0f; m[1] = 0.0f; m[2] = 0.0f;
        m[3] = 0.0f; m[4] = 1.0f; m[5] = 0.0f;
        m[6] = 0.0f; m[7] = 0.0f; m[8] = 1.0f;
    }

    // Explicit 9-element constructor
    __host__ __device__ mat3(
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22
    ) {
        m[0] = m00; m[1] = m01; m[2] = m02;
        m[3] = m10; m[4] = m11; m[5] = m12;
        m[6] = m20; m[7] = m21; m[8] = m22;
    }

    // Element access (row, col)
    __host__ __device__ float operator()(int row, int col) const {
        return m[row * 3 + col];
    }

    __host__ __device__ float& operator()(int row, int col) {
        return m[row * 3 + col];
    }

    // Matrix * Vector multiplication
    __host__ __device__ vec3 operator*(const vec3& v) const {
        return vec3(
            m[0] * v.x() + m[1] * v.y() + m[2] * v.z(),
            m[3] * v.x() + m[4] * v.y() + m[5] * v.z(),
            m[6] * v.x() + m[7] * v.y() + m[8] * v.z()
        );
    }

    __host__ __device__ mat3 operator*(const mat3& other) const {
        mat3 result;

        result.m[0] = m[0] * other.m[0] + m[1] * other.m[3] + m[2] * other.m[6];
        result.m[1] = m[0] * other.m[1] + m[1] * other.m[4] + m[2] * other.m[7];
        result.m[2] = m[0] * other.m[2] + m[1] * other.m[5] + m[2] * other.m[8];

        result.m[3] = m[3] * other.m[0] + m[4] * other.m[3] + m[5] * other.m[6];
        result.m[4] = m[3] * other.m[1] + m[4] * other.m[4] + m[5] * other.m[7];
        result.m[5] = m[3] * other.m[2] + m[4] * other.m[5] + m[5] * other.m[8];

        result.m[6] = m[6] * other.m[0] + m[7] * other.m[3] + m[8] * other.m[6];
        result.m[7] = m[6] * other.m[1] + m[7] * other.m[4] + m[8] * other.m[7];
        result.m[8] = m[6] * other.m[2] + m[7] * other.m[5] + m[8] * other.m[8];

        return result;
    }

    // Transpose (which equals inverse for pure rotation)
    __host__ __device__ mat3 transpose() const {
        return mat3(
            m[0], m[3], m[6],
            m[1], m[4], m[7],
            m[2], m[5], m[8]
        );
    }

    // Static rotation builders
    __host__ __device__ static mat3 rotate_x(float rad) {
        float c = cosf(rad), s = sinf(rad);
        return mat3(1, 0,  0,
                    0, c, -s,
                    0, s,  c);
    }

    __host__ __device__ static mat3 rotate_y(float rad) {
        float c = cosf(rad), s = sinf(rad);
        return mat3( c, 0, s,
                     0, 1, 0,
                    -s, 0, c);
    }

    __host__ __device__ static mat3 rotate_z(float rad) {
        float c = cosf(rad), s = sinf(rad);
        return mat3(c, -s, 0,
                    s,  c, 0,
                    0,  0, 1);
    }
};

#endif