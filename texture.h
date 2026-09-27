#ifndef TEXTURE_H
#define TEXTURE_H

#include "rtweekend.cuh"


class texture {
  public:
    __device__ virtual ~texture() = default;

    __device__ virtual colour value(float u, float v, const point3& p) const = 0;
};

class solid_colour : public texture {
  public:
    __device__ solid_colour(const colour& albedo) : albedo(albedo) {}

    __device__ solid_colour(float red, float green, float blue) : solid_colour(colour(red,green,blue)) {}

    __device__ colour value(float u, float v, const point3& p) const override {
        return albedo;
    }

  private:
    colour albedo;
};

class checker_texture : public texture {
    public:
        __device__ checker_texture(float scale, texture *even, texture *odd) : scale(scale), even(even), odd(odd) {}

        __device__ colour value(float u, float v, const point3& p) const override {
            const int u_integer = static_cast<int>(floorf(scale * u));
            const int v_integer = static_cast<int>(floorf(scale * v));

            const bool is_even = ((u_integer + v_integer) & 1) == 0;
            return is_even ? even->value(u, v, p) : odd->value(u, v, p);
        }

    private:
        float scale;
        texture* even;
        texture* odd;

};

class image_texture : public texture {
public:
    __device__ image_texture() 
        : pixels(nullptr), width(0), height(0), bytes_per_pixel(3), bytes_per_scanline(0) {}

    __device__ image_texture(const unsigned char* d_pixels, int w, int h, int bpp = 3)
        : pixels(d_pixels), width(w), height(h), bytes_per_pixel(bpp),
          bytes_per_scanline(w * bpp) {}

__device__ colour value(
    float u,
    float v,
    const point3&
) const override {
    if (pixels == nullptr || width <= 0 || height <= 0) {
        return colour(0.0f, 1.0f, 1.0f);
    }

    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);

    int i = static_cast<int>(u * width);
    int j = static_cast<int>(v * height);

    i = min(i, width - 1);
    j = min(j, height - 1);

    const unsigned char* pixel =
        pixels
        + j * bytes_per_scanline
        + i * bytes_per_pixel;

    constexpr float scale = 1.0f / 255.0f;

    return colour(
        scale * pixel[0],
        scale * pixel[1],
        scale * pixel[2]
    );
}

private:
    const unsigned char* pixels;
    int width;
    int height;
    int bytes_per_pixel;
    int bytes_per_scanline;
};

#endif
