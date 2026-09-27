#ifndef IMAGE_H
#define IMAGE_H

#include <cuda_runtime.h>

#include <string>


class image {
public:
    image() = default;

    explicit image(const char* image_filename);

    ~image();

    image(const image&) = delete;
    image& operator=(const image&) = delete;

    image(image&& other) noexcept;
    image& operator=(image&& other) noexcept;


    bool load(const std::string& filename);


    int width() const {
        return image_width;
    }


    int height() const {
        return image_height;
    }


    int get_bytes_per_pixel() const {
        return bytes_per_pixel;
    }


    const unsigned char* pixel_data(int x, int y) const;


    bool copy_to_device(
        unsigned char** d_pixels,
        int* d_width = nullptr,
        int* d_height = nullptr
    ) const;


private:
    static constexpr int bytes_per_pixel = 3;

    unsigned char* bdata = nullptr;

    int image_width = 0;
    int image_height = 0;
    int bytes_per_scanline = 0;


    void clear();


    static int clamp(int value, int low, int high);
};


#endif