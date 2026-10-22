#ifndef TEXTURE_H
#define TEXTURE_H

#include "device_math.cuh"

enum class TextureType : unsigned char {
    SolidColour,
    Checker,
    Image
};

struct texture;

// Plain payload types keep the union straightforward.
// Store solid-colour components directly rather than putting
// a potentially non-trivial vec3/colour object inside the union.

struct SolidColourTextureData {
    float red;
    float green;
    float blue;
};

struct CheckerTextureData {
    float scale;
    const texture* even;
    const texture* odd;
};

struct ImageTextureData {
    const unsigned char* pixels;
    int width;
    int height;
    int bytes_per_pixel;
    int bytes_per_scanline;
};

union TextureData {
    SolidColourTextureData solid;
    CheckerTextureData checker;
    ImageTextureData image;
};

struct texture {
public:
    // Matches the old default image_texture:
    // an invalid image evaluates to cyan.
    __device__ texture()
        : type(TextureType::Image), data{} {
        data.image = ImageTextureData{
            nullptr,
            0,
            0,
            3,
            0
        };
    }

    // Solid colour from a colour/vec3.
    __device__ explicit texture(const colour& albedo)
        : type(TextureType::SolidColour), data{} {
        data.solid = SolidColourTextureData{
            albedo.x(),
            albedo.y(),
            albedo.z()
        };
    }

    // Solid colour from three components.
    __device__ texture(float red, float green, float blue)
        : type(TextureType::SolidColour), data{} {
        data.solid = SolidColourTextureData{
            red,
            green,
            blue
        };
    }

    // Checker texture.
    __device__ texture(
        float scale,
        const texture* even,
        const texture* odd
    )
        : type(TextureType::Checker), data{} {
        data.checker = CheckerTextureData{
            scale,
            even,
            odd
        };
    }

    // Image texture.
    __device__ texture(
        const unsigned char* d_pixels,
        int width,
        int height,
        int bytes_per_pixel = 3
    )
        : type(TextureType::Image), data{} {
        data.image = ImageTextureData{
            d_pixels,
            width,
            height,
            bytes_per_pixel,
            width * bytes_per_pixel
        };
    }

    // No virtual functions and no recursive checker calls.
    __device__ __forceinline__
    colour value(float u, float v, const point3& p) const {
        // Your existing textures do not use p.
        (void)p;

        const texture* current = this;

        while (current != nullptr) {
            switch (current->type) {
                case TextureType::SolidColour: {
                    const SolidColourTextureData& solid =
                        current->data.solid;

                    return colour(
                        solid.red,
                        solid.green,
                        solid.blue
                    );
                }

                case TextureType::Checker: {
                    const CheckerTextureData& checker =
                        current->data.checker;

                    const int u_integer =
                        static_cast<int>(floorf(checker.scale * u));

                    const int v_integer =
                        static_cast<int>(floorf(checker.scale * v));

                    const bool is_even =
                        ((u_integer + v_integer) & 1) == 0;

                    // Follow the selected child and dispatch again.
                    // This also supports checkers containing checkers.
                    current = is_even ? checker.even : checker.odd;
                    break;
                }

                case TextureType::Image:
                    return sample_image(current->data.image, u, v);

                default:
                    return colour(0.0f, 1.0f, 1.0f);
            }
        }

        // Diagnostic colour for a null checker child.
        return colour(0.0f, 1.0f, 1.0f);
    }

private:
    TextureType type;
    TextureData data;

    __device__ __forceinline__
    static colour sample_image(
        const ImageTextureData& image,
        float u,
        float v
    ) {
        if (
            image.pixels == nullptr ||
            image.width <= 0 ||
            image.height <= 0 ||
            image.bytes_per_pixel < 3
        ) {
            return colour(0.0f, 1.0f, 1.0f);
        }

        u = fminf(fmaxf(u, 0.0f), 1.0f);
        v = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);

        int i = static_cast<int>(u * image.width);
        int j = static_cast<int>(v * image.height);

        i = min(i, image.width - 1);
        j = min(j, image.height - 1);

        const unsigned char* pixel =
            image.pixels
            + j * image.bytes_per_scanline
            + i * image.bytes_per_pixel;

        constexpr float scale = 1.0f / 255.0f;

        return colour(
            scale * pixel[0],
            scale * pixel[1],
            scale * pixel[2]
        );
    }
};

// Compatibility names: these are all the same concrete type,
// not subclasses of texture.
using solid_colour = texture;
using checker_texture = texture;
using image_texture = texture;

#endif