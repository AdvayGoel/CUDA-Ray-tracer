#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#ifdef _MSC_VER
    #pragma warning(push, 0)
#endif

#include "external/stb_image.h"

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "image.h"

#include <cstdlib>
#include <iostream>
#include <utility>


image::image(const char* image_filename) {
    if (image_filename == nullptr) {
        std::cerr << "ERROR: Image filename was null.\n";
        return;
    }

    // searches for the image file
    const std::string filename(image_filename);

    const char* image_directory = std::getenv("IMAGES");

    if (image_directory != nullptr
        && load(std::string(image_directory) + "/" + filename)) {
        return;
    }

    if (load(filename)) {
        return;
    }

    if (load("images/" + filename)) {
        return;
    }

    if (load("../images/" + filename)) {
        return;
    }

    if (load("../../images/" + filename)) {
        return;
    }

    if (load("../../../images/" + filename)) {
        return;
    }

    if (load("../../../../images/" + filename)) {
        return;
    }

    if (load("../../../../../images/" + filename)) {
        return;
    }

    if (load("../../../../../../images/" + filename)) {
        return;
    }

    std::cerr
        << "ERROR: Could not load image file '"
        << image_filename
        << "'";

    const char* failure_reason = stbi_failure_reason();

    if (failure_reason != nullptr) {
        std::cerr << ": " << failure_reason;
    }

    std::cerr << ".\n";
}


image::~image() {
    clear();
}


image::image(image&& other) noexcept
    : bdata(other.bdata),
      image_width(other.image_width),
      image_height(other.image_height),
      bytes_per_scanline(other.bytes_per_scanline) {
    other.bdata = nullptr;
    other.image_width = 0;
    other.image_height = 0;
    other.bytes_per_scanline = 0;
}


image& image::operator=(image&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    clear();

    bdata = other.bdata;
    image_width = other.image_width;
    image_height = other.image_height;
    bytes_per_scanline = other.bytes_per_scanline;

    other.bdata = nullptr;
    other.image_width = 0;
    other.image_height = 0;
    other.bytes_per_scanline = 0;

    return *this;
}


void image::clear() {
    if (bdata != nullptr) {
        stbi_image_free(bdata);
        bdata = nullptr;
    }

    image_width = 0;
    image_height = 0;
    bytes_per_scanline = 0;
}


bool image::load(const std::string& filename) {
    clear();

    int loaded_width = 0;
    int loaded_height = 0;
    int channels_in_file = 0;

    unsigned char* loaded_pixels = stbi_load(
        filename.c_str(),
        &loaded_width,
        &loaded_height,
        &channels_in_file,
        bytes_per_pixel
    );

    if (loaded_pixels == nullptr) {
        return false;
    }

    if (loaded_width <= 0 || loaded_height <= 0) {
        stbi_image_free(loaded_pixels);
        return false;
    }

    bdata = loaded_pixels;
    image_width = loaded_width;
    image_height = loaded_height;
    bytes_per_scanline = image_width * bytes_per_pixel;

    return true;
}


const unsigned char* image::pixel_data(int x, int y) const {
    static const unsigned char magenta[] = {255, 0, 255};

    if (bdata == nullptr || image_width <= 0 || image_height <= 0) {
        return magenta;
    }

    x = clamp(x, 0, image_width);
    y = clamp(y, 0, image_height);

    return bdata
        + y * bytes_per_scanline
        + x * bytes_per_pixel;
}

// copies the image and associated metadata from HOST memory to DEVICE memory
bool image::copy_to_device(
    unsigned char** d_pixels,
    int* d_width,
    int* d_height
) const {
    if (d_pixels == nullptr
        || bdata == nullptr
        || image_width <= 0
        || image_height <= 0) {
        return false;
    }

    *d_pixels = nullptr;

    const size_t total_bytes =
        static_cast<size_t>(image_width)
        * static_cast<size_t>(image_height)
        * static_cast<size_t>(bytes_per_pixel);

    unsigned char* d_ptr = nullptr;

    cudaError_t result = cudaMalloc(&d_ptr, total_bytes);

    if (result != cudaSuccess) {
        return false;
    }

    result = cudaMemcpy(
        d_ptr,
        bdata,
        total_bytes,
        cudaMemcpyHostToDevice
    );

    if (result != cudaSuccess) {
        cudaFree(d_ptr);
        return false;
    }

    *d_pixels = d_ptr;

    if (d_width != nullptr) {
        *d_width = image_width;
    }

    if (d_height != nullptr) {
        *d_height = image_height;
    }

    return true;
}


int image::clamp(int value, int low, int high) {
    if (value < low) {
        return low;
    }

    if (value >= high) {
        return high - 1;
    }

    return value;
}