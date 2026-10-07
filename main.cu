#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "device_math.cuh"
#include "colour.h"
#include "texture.h"
#include "material.cuh"
#include "hittable.h"
#include "sphere.h"
#include "cuboid.h"
#include "quad.h"
#include "transform.cuh"
#include "camera.cuh"
#include "bvh.cuh"
#include "image.h"

#include <chrono>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(
    cudaError_t result,
    const char* function,
    const char* file,
    int line
) {
    if (result == cudaSuccess) {
        return;
    }

    std::cerr
        << "CUDA error: " << cudaGetErrorString(result)
        << " (" << static_cast<int>(result) << ")\n"
        << "  Expression: " << function << "\n"
        << "  Location: " << file << ":" << line << "\n";

    cudaDeviceReset();
    std::exit(EXIT_FAILURE);
}


// ----------------------------------------------------------------------------
// GENERIC IMAGE METADATA CONTAINER
// Transferred from host to device so kernels have image dimensions & pointers
// ----------------------------------------------------------------------------
struct ImageMetadata {
    unsigned char* pixels;
    int width;
    int height;
    int bytes_per_pixel;
};


// ----------------------------------------------------------------------------
// HOST PROXY FOR BVH CONSTRUCTION
// ----------------------------------------------------------------------------
class HostProxyPrimitive : public hittable {
public:
    int id;
    aabb box;

    HostProxyPrimitive(int id, const aabb& box)
        : id(id), box(box) {}

    __device__ bool hit(
        const ray&,
        interval,
        hit_record&
    ) const override {
        return false;
    }

    __device__ __host__ aabb bounding_box() const override {
        return box;
    }
};


// ----------------------------------------------------------------------------
// SCENE SETUP FUNCTIONS (CALLED FROM KERNEL)
// ----------------------------------------------------------------------------
__device__ void create_light_box(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;


    // Textures
    d_textures[tex_idx++] = new solid_colour(
        colour(0.85f, 0.0f, 0.0f)
    );


    // Materials
    d_materials[mat_idx++] = new diffuse_light(
        colour(4.0f, 4.0f, 4.0f)
    );
    d_materials[mat_idx++] = new lambertian(d_textures[0]);

    // Primitives
    cuboid *light = new cuboid(point3(0.5, 0.5, 0.5), 0.5, 0.5, 0.5, d_materials[0]);

    d_raw_list[obj_idx++] = light;
    d_raw_list[obj_idx++] = new sphere(
        point3(0.0f, -1000.0f, 0.0f),
        1000.0f,
        d_materials[1]
    );

    d_lights[light_idx++] = light;


    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = d_raw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *light_count = light_idx;
}


__device__ void create_snowman_kernel(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    unsigned int seed = 1337u;

    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;


    // Base textures for Lambertian materials
    d_textures[tex_idx++] = new solid_colour(
        colour(0.85f, 0.88f, 0.92f)
    ); // 0: ground

    d_textures[tex_idx++] = new solid_colour(
        colour(0.92f, 0.93f, 0.96f)
    ); // 1: snow

    d_textures[tex_idx++] = new solid_colour(
        colour(0.35f, 0.20f, 0.10f)
    ); // 2: wood

    d_textures[tex_idx++] = new solid_colour(
        colour(0.05f, 0.05f, 0.05f)
    ); // 3: coal


    // Materials
    d_materials[mat_idx++] = new lambertian(d_textures[0]); // 0: ground
    d_materials[mat_idx++] = new lambertian(d_textures[1]); // 1: snow
    d_materials[mat_idx++] = new metal(
        colour(0.12f, 0.12f, 0.12f),
        0.1f
    ); // 2: hat

    d_materials[mat_idx++] = new lambertian(d_textures[2]); // 3: wood

    d_materials[mat_idx++] = new diffuse_light(
        colour(0.95f, 0.45f, 0.05f)
    ); // 4: carrot

    d_materials[mat_idx++] = new lambertian(d_textures[3]); // 5: coal


    // 1. Snowy ground
    d_raw_list[obj_idx++] = new sphere(
        point3(0.0f, -1000.0f, 0.0f),
        1000.0f,
        d_materials[0]
    );


    // 2. Snowman body (3-tier stack)
    d_raw_list[obj_idx++] = new sphere(
        point3(0.0f, 1.2f, 0.0f),
        1.2f,
        d_materials[1]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.0f, 2.7f, 0.0f),
        0.8f,
        d_materials[1]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.0f, 3.8f, 0.0f),
        0.5f,
        d_materials[1]
    );


    // 3. Top Hat
    d_raw_list[obj_idx++] = new cuboid(
        point3(0.0f, 4.25f, 0.0f),
        1.30f,
        0.10f,
        1.30f,
        d_materials[2]
    );

    d_raw_list[obj_idx++] = new cuboid(
        point3(0.0f, 4.65f, 0.0f),
        0.80f,
        0.70f,
        0.80f,
        d_materials[2]
    );


    // 4. Arms using Prototypes + Transform Instancing
    // A single base arm branch centered at origin:

    d_prototypes[proto_idx++] = new cuboid(
        point3(0.0f, 0.0f, 0.55f),
        0.12f,
        0.12f,
        1.10f,
        d_materials[3]
    );


    // Instanced Right Arm
    d_raw_list[obj_idx++] = new transform(
        d_prototypes[0],
        vec3(0.0f, 2.70f, 0.75f),
        degrees_to_radians(-20.0f), // Pitch / elevation
        degrees_to_radians(15.0f),  // Yaw / angle forward
        0.0f
    );



    // Instanced Left Arm
    d_raw_list[obj_idx++] = new transform(
        d_prototypes[0],
        vec3(0.0f, 2.70f, -0.75f),
        degrees_to_radians(-20.0f),
        degrees_to_radians(180.0f - 15.0f), // Flip direction along -Z
        0.0f
    );


    // 5. Carrot nose & coal features
    d_raw_list[obj_idx++] = new cuboid(
        point3(0.67f, 3.80f, 0.0f),
        0.45f,
        0.10f,
        0.12f,
        d_materials[4]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.45f, 3.98f, 0.18f),
        0.07f,
        d_materials[5]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.45f, 3.98f, -0.18f),
        0.07f,
        d_materials[5]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.78f, 2.95f, 0.00f),
        0.08f,
        d_materials[5]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.80f, 2.70f, 0.00f),
        0.08f,
        d_materials[5]
    );

    d_raw_list[obj_idx++] = new sphere(
        point3(0.75f, 2.45f, 0.00f),
        0.08f,
        d_materials[5]
    );


    // 6. Background procedural spheres
    for (int a = -30; a <= 30; a += 2) {
        for (int b = -30; b <= 30; b += 2) {
            const float dist_to_snowman = sqrtf(
                static_cast<float>(a * a + b * b)
            );

            if (dist_to_snowman > 2.5f) {
                const point3 pos(
                    static_cast<float>(a)
                        + 0.6f * rand_float(&seed),
                    0.25f,
                    static_cast<float>(b)
                        + 0.6f * rand_float(&seed)
                );

                material* mat = nullptr;

                if (rand_float(&seed) < 0.5f) {
                    mat = new dielectric(1.31f);
                } else {
                    mat = new metal(
                        colour(0.8f, 0.85f, 0.9f),
                        0.05f
                    );
                }

                d_materials[mat_idx++] = mat;
                d_raw_list[obj_idx++] = new sphere(
                    pos,
                    0.25f,
                    mat
                );
            }
        }
    }


    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = d_raw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *light_count = light_idx;
}


__device__ void create_quads(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;

    // Textures
    d_textures[tex_idx++] = new solid_colour(
        colour(0.65f, 0.05f, 0.05f)
    ); // 0: red

    d_textures[tex_idx++] = new solid_colour(
        colour(0.73f, 0.73f, 0.73f)
    ); // 1: white

    d_textures[tex_idx++] = new solid_colour(
        colour(0.12f, 0.45f, 0.15f)
    ); // 2: green

    d_textures[tex_idx++] = new checker_texture(
        10.0f,
        d_textures[0],
        d_textures[2]
    ); // 3: checker


    // Materials
    d_materials[mat_idx++] = new lambertian(d_textures[0]); // 0: red
    d_materials[mat_idx++] = new lambertian(d_textures[1]); // 1: white
    d_materials[mat_idx++] = new lambertian(d_textures[2]); // 2: green
    d_materials[mat_idx++] = new diffuse_light(
        colour(15.0f, 15.0f, 15.0f)
    ); // 3: light
    d_materials[mat_idx++] = new lambertian(d_textures[3]); // 4: checker


    // Walls (Cornell Box)
    d_raw_list[obj_idx++] = new quad(
        point3(555.0f, 0.0f, 0.0f),
        vec3(0.0f, 555.0f, 0.0f),
        vec3(0.0f, 0.0f, 555.0f),
        d_materials[2]
    ); // left wall (green)

    d_raw_list[obj_idx++] = new quad(
        point3(0.0f, 0.0f, 0.0f),
        vec3(0.0f, 555.0f, 0.0f),
        vec3(0.0f, 0.0f, 555.0f),
        d_materials[0]
    ); // right wall (red)

    quad* light = new quad(
        point3(343.0f, 554.0f, 332.0f),
        vec3(-130.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, -105.0f),
        d_materials[3]
    ); // ceiling light

    d_raw_list[obj_idx++] = light;

    d_lights[light_idx++] = light;

    d_raw_list[obj_idx++] = new quad(
        point3(0.0f, 0.0f, 0.0f),
        vec3(555.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, 555.0f),
        d_materials[1]
    ); // floor

    d_raw_list[obj_idx++] = new quad(
        point3(0.0f, 555.0f, 0.0f),
        vec3(555.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, 555.0f),
        d_materials[1]
    ); // ceiling

    d_raw_list[obj_idx++] = new quad(
        point3(0.0f, 0.0f, 555.0f),
        vec3(555.0f, 0.0f, 0.0f),
        vec3(0.0f, 555.0f, 0.0f),
        d_materials[1]
    ); // back wall


    // Base box prototype centered at origin
    d_prototypes[proto_idx++] = new cuboid(
        point3(0.0f, 0.0f, 0.0f),
        200.0f,
        300.0f,
        200.0f,
        d_materials[4]
    );


    // Place the rotated instance into the scene
    d_raw_list[obj_idx++] = new transform(
        d_prototypes[0],
        vec3(278.0f, 150.0f, 278.0f),
        0.0f,
        degrees_to_radians(45.0f),
        0.0f,
        d_materials[1]
    );


    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = d_raw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *light_count = light_idx;
}

__device__ void create_earth_scene(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;

    const ImageMetadata& earth_meta = d_scene_images[0];

    d_textures[tex_idx++] = new image_texture(
        earth_meta.pixels,
        earth_meta.width,
        earth_meta.height,
        earth_meta.bytes_per_pixel
    );

    d_materials[mat_idx++] = new lambertian(d_textures[0]);

    d_raw_list[obj_idx++] = new sphere(
        point3(0.0f, 0.0f, 0.0f),
        2.0f,
        d_materials[0]
    );

    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = d_raw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *light_count = light_idx;
}


__device__ void create_table_earth_scene(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;


    // --------------------------------------------------------------------
    // Textures
    // --------------------------------------------------------------------
    // 0: Earth map from host image buffer
    const ImageMetadata& earth_meta = d_scene_images[0];
    const ImageMetadata& wooden_meta = d_scene_images[1];

    d_textures[tex_idx++] = new image_texture(
        earth_meta.pixels,
        earth_meta.width,
        earth_meta.height,
        earth_meta.bytes_per_pixel
    );

    // 1: Warm wood for the tabletop
    d_textures[tex_idx++] = new solid_colour(
        colour(0.38f, 0.26f, 0.19f)
    );

    // 2: Neutral wall plaster
    d_textures[tex_idx++] = new solid_colour(
        colour(0.82f, 0.82f, 0.79f)
    );

    // 3: Wooden floor texture
    d_textures[tex_idx++] = new image_texture(
        wooden_meta.pixels,
        wooden_meta.width,
        wooden_meta.height,
        wooden_meta.bytes_per_pixel
    );


    // --------------------------------------------------------------------
    // Materials
    // --------------------------------------------------------------------
    d_materials[mat_idx++] = new lambertian(d_textures[0]); // 0: Earth
    d_materials[mat_idx++] = new lambertian(d_textures[1]); // 1: Wood Table
    d_materials[mat_idx++] = new lambertian(d_textures[2]); // 2: Walls & Ceiling
    d_materials[mat_idx++] = new lambertian(d_textures[3]); // 3: Wooden tiles

    d_materials[mat_idx++] = new diffuse_light(
        colour(1.9f, 1.8f, 1.6f)
    ); // 4: Soft Room Light

    d_materials[mat_idx++] = new diffuse_light(
        colour(1.0f, 1.0f, 1.1f)
    ); // 5: Foreground Key Light

    d_materials[mat_idx++] = new dielectric(1.5f); // 6: Glass


    // Earth prototype
    d_prototypes[proto_idx++] = new sphere(
        point3(0.0f, 0.0f, 0.0f),
        0.22f,
        d_materials[0]
    );


    // --------------------------------------------------------------------
    // 1. The Large Room (Dimensions: 14m wide, 5m high, 14m deep)
    // Floor sits at y = 0.0, ceiling at y = 5.0
    // --------------------------------------------------------------------
    // Floor
    d_raw_list[obj_idx++] = new quad(
        point3(-7.0f, 0.0f, -11.0f),
        vec3(14.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, 14.0f),
        d_materials[1]
    );

    // Ceiling
    d_raw_list[obj_idx++] = new quad(
        point3(-7.0f, 5.0f, -11.0f),
        vec3(14.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, 14.0f),
        d_materials[2]
    );

    // Back Wall
    // ----------------------------------------------------------------------------
    // BACK WALL WITH WINDOW CUTOUT & THIN GLASS PANE
    // Wall: X in [-7, 7], Y in [0, 5], Z = -11.0
    // Window opening: X in [-3, 3], Y in [1.0, 3.8] (6m wide, 2.8m high opening)
    // ----------------------------------------------------------------------------

    // 1. Bottom wall panel (Y: 0.0 to 1.0)
    d_raw_list[obj_idx++] = new quad(
        point3(-7.0f, 0.0f, -11.0f),
        vec3(14.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f),
        d_materials[2]
    );

    // 2. Top wall panel (Y: 3.8 to 5.0)
    d_raw_list[obj_idx++] = new quad(
        point3(-7.0f, 3.8f, -11.0f),
        vec3(14.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.2f, 0.0f),
        d_materials[2]
    );

    // 3. Left wall panel (X: -7.0 to -3.0, Y: 1.0 to 3.8)
    d_raw_list[obj_idx++] = new quad(
        point3(-7.0f, 1.0f, -11.0f),
        vec3(4.0f, 0.0f, 0.0f),
        vec3(0.0f, 2.8f, 0.0f),
        d_materials[2]
    );

    // 4. Right wall panel (X: 3.0 to 7.0, Y: 1.0 to 3.8)
    d_raw_list[obj_idx++] = new quad(
        point3(3.0f, 1.0f, -11.0f),
        vec3(4.0f, 0.0f, 0.0f),
        vec3(0.0f, 2.8f, 0.0f),
        d_materials[2]
    );

    // 5. Thin Glass Window Pane (Width = 5.98m, Height = 2.78m, Thickness = 0.02m)
    // Center: (0.0f, 2.4f, -11.0f). Extends from Z = -11.01 to -10.99
    // Fits flush inside the wall opening so side faces are hidden inside the wall!
    d_raw_list[obj_idx++] = new cuboid(
        point3(0.0f, 2.4f, -11.0f),
        5.98f,
        2.78f,
        0.02f,
        d_materials[6]
    );

    // Left Wall
    d_raw_list[obj_idx++] = new quad(
        point3(-7.0f, 0.0f, -11.0f),
        vec3(0.0f, 0.0f, 14.0f),
        vec3(0.0f, 5.0f, 0.0f),
        d_materials[2]
    );

    // Right Wall
    d_raw_list[obj_idx++] = new quad(
        point3(7.0f, 0.0f, -11.0f),
        vec3(0.0f, 0.0f, 14.0f),
        vec3(0.0f, 5.0f, 0.0f),
        d_materials[2]
    );

    // Overhead Area Light (diffuse panel on ceiling)
    quad* light_1 = new quad(
        point3(-2.0f, 4.98f, -6.0f),
        vec3(4.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, 4.0f),
        d_materials[4]
    );

    d_raw_list[obj_idx++] = light_1;
    d_lights[light_idx++] = light_1;

    // Soft Key Light above camera to fill foreground Earth details
    quad* light_2 = new quad(
        point3(-1.0f, 3.5f, 2.0f),
        vec3(2.0f, 0.0f, 0.0f),
        vec3(0.0f, 0.0f, -1.5f),
        d_materials[5]
    );

    d_raw_list[obj_idx++] = light_2;
    d_lights[light_idx++] = light_2;

    // --------------------------------------------------------------------
    // 2. The Table (Foreground Right)
    // Tabletop top surface is at y = 1.0. Corner sits at (x = 0.15, z = -1.30)
    // --------------------------------------------------------------------
    // Table top: center (-0.95, 0.96, -0.2), dimensions (2.2, 0.08, 2.2)
    // Bounds: X [-2.05, 0.15], Y [0.92, 1.00], Z [-1.30, 0.90]
    d_raw_list[obj_idx++] = new cuboid(
        point3(-0.95f, 0.96f, -0.20f),
        2.20f,
        0.08f,
        2.20f,
        d_materials[3]
    );


    // Table leg near the visible corner
    d_raw_list[obj_idx++] = new cuboid(
        point3(0.05f, 0.46f, -1.20f),
        0.1f,
        0.92f,
        0.1f,
        d_materials[1]
    );

    d_raw_list[obj_idx++] = new transform(
        d_raw_list[obj_idx - 1],
        vec3(-2.0f, 0.0f, 0.0f),
        0.0f,
        0.0f,
        0.0f
    );

    d_raw_list[obj_idx++] = new transform(
        d_raw_list[obj_idx - 2],
        vec3(-2.0f, 0.0f, 2.0f),
        0.0f,
        0.0f,
        0.0f
    );

    d_raw_list[obj_idx++] = new transform(
        d_raw_list[obj_idx - 3],
        vec3(0.0f, 0.0f, 2.0f),
        0.0f,
        0.0f,
        0.0f
    );

    d_raw_list[obj_idx++] = new cuboid(
        point3(-1.95f, 0.46f, -1.20f),
        0.1f,
        0.92f,
        0.1f,
        d_materials[1]
    );


    // --------------------------------------------------------------------
    // 3. Small Earth Sphere (Foreground Edge)
    // Radius = 0.22m, resting on y = 1.00 -> Center at y = 1.22
    // Placed right near the table corner at (-1.90, 1.22, -1.15)
    // --------------------------------------------------------------------
    d_raw_list[obj_idx++] = new transform(
        d_prototypes[0],
        point3(-1.90f, 1.22f, -1.15f),
        0.0f,
        degrees_to_radians(180.0f),
        0.0f
    );


    // --------------------------------------------------------------------
    // 4. Background Accents (Giving the large room scale and depth)
    // -------------------------------------------------------------------
    // Compute bounding boxes for BVH builder
    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = d_raw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *light_count = light_idx;
}

__device__ void create_spheres_benchmark_kernel(
    hittable** draw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* d_light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int seed = 42u; // Fixed deterministic seed for reproducible benchmarks
    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;

    // ------------------------------------------------------------------------
    // 1. Ground Plane (Giant Sphere)
    // ------------------------------------------------------------------------
    d_textures[tex_idx] = new solid_colour(colour(0.5f, 0.5f, 0.5f));
    d_materials[mat_idx] = new lambertian(d_textures[tex_idx]);
    tex_idx++;
    mat_idx++;

    draw_list[obj_idx] = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, d_materials[0]);
    obj_idx++;

    // ------------------------------------------------------------------------
    // 2. Three Central Hero Spheres
    // ------------------------------------------------------------------------
    // Glass sphere
    d_materials[mat_idx] = new dielectric(1.5f);
    draw_list[obj_idx] = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, d_materials[mat_idx]);
    mat_idx++;
    obj_idx++;

    // Matte diffuse sphere
    d_textures[tex_idx] = new solid_colour(colour(0.4f, 0.2f, 0.1f));
    d_materials[mat_idx] = new lambertian(d_textures[tex_idx]);
    tex_idx++;
    draw_list[obj_idx] = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, d_materials[mat_idx]);
    mat_idx++;
    obj_idx++;

    // Smooth metal sphere
    d_materials[mat_idx] = new metal(colour(0.7f, 0.6f, 0.5f), 0.0f);
    draw_list[obj_idx] = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, d_materials[mat_idx]);
    mat_idx++;
    obj_idx++;

    // ------------------------------------------------------------------------
    // 3. Grid of Random Small Spheres (Target: ~1,000 Total Primitives)
    // Range: a in [-17, 17], b in [-14, 15] -> 35 * 30 = 1050 candidate cells
    // ------------------------------------------------------------------------
    for (int a = -17; a <= 17; ++a) {
        for (int b = -14; b <= 15; ++b) {
            if (obj_idx >= 1000) break;

            const float choose_mat = rand_float(&seed);
            const point3 center(
                static_cast<float>(a) + 0.9f * rand_float(&seed),
                0.2f,
                static_cast<float>(b) + 0.9f * rand_float(&seed)
            );

            // Avoid placing small spheres inside the 3 large hero spheres
            const float dist_diffuse = (center - point3(-4.0f, 1.0f, 0.0f)).length();
            const float dist_glass   = (center - point3( 0.0f, 1.0f, 0.0f)).length();
            const float dist_metal   = (center - point3( 4.0f, 1.0f, 0.0f)).length();

            if (dist_diffuse > 1.25f && dist_glass > 1.25f && dist_metal > 1.25f) {
                material* mat = nullptr;

                if (choose_mat < 0.70f) {
                    // Diffuse
                    const colour albedo(
                        rand_float(&seed) * rand_float(&seed),
                        rand_float(&seed) * rand_float(&seed),
                        rand_float(&seed) * rand_float(&seed)
                    );
                    d_textures[tex_idx] = new solid_colour(albedo);
                    mat = new lambertian(d_textures[tex_idx]);
                    tex_idx++;
                } else if (choose_mat < 0.88f) {
                    // Metal
                    const colour albedo(
                        0.5f * (1.0f + rand_float(&seed)),
                        0.5f * (1.0f + rand_float(&seed)),
                        0.5f * (1.0f + rand_float(&seed))
                    );
                    const float fuzz = 0.5f * rand_float(&seed);
                    mat = new metal(albedo, fuzz);
                } else {
                    // Glass
                    mat = new dielectric(1.5f);
                }

                d_materials[mat_idx] = mat;
                mat_idx++;

                draw_list[obj_idx] = new sphere(center, 0.2f, mat);
                obj_idx++;
            }
        }
    }

    // Compute bounding boxes for the Host BVH builder
    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = draw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *d_light_count = light_idx;
}

__device__ void create_cornell_glass_kernel(
    hittable** draw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* d_light_count,
    const ImageMetadata* d_scene_images
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int mat_idx = 0;
    int tex_idx = 0;
    int proto_idx = 0;
    int obj_idx = 0;
    int light_idx = 0;

    // Textures
    d_textures[tex_idx++] = new solid_colour(colour(0.65f, 0.05f, 0.05f)); // 0: Red
    d_textures[tex_idx++] = new solid_colour(colour(0.73f, 0.73f, 0.73f)); // 1: White
    d_textures[tex_idx++] = new solid_colour(colour(0.12f, 0.45f, 0.15f)); // 2: Green

    // Materials
    d_materials[mat_idx++] = new lambertian(d_textures[0]);                       // 0: Red
    d_materials[mat_idx++] = new lambertian(d_textures[1]);                       // 1: White
    d_materials[mat_idx++] = new lambertian(d_textures[2]);                       // 2: Green
    d_materials[mat_idx++] = new diffuse_light(colour(15.0f, 15.0f, 15.0f));     // 3: Light
    d_materials[mat_idx++] = new dielectric(1.5f);                                // 4: Glass (dielectric)
    d_materials[mat_idx++] = new metal(colour(0.85f, 0.85f, 0.85f), 0.0f);        // 5: Mirror metal

    // Walls
    draw_list[obj_idx++] = new quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_materials[2]); // Green
    draw_list[obj_idx++] = new quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_materials[0]);   // Red
    draw_list[obj_idx++] = new quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), d_materials[1]);   // Floor
    draw_list[obj_idx++] = new quad(point3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), d_materials[1]); // Ceiling
    draw_list[obj_idx++] = new quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), d_materials[1]); // Back

    // Ceiling Light
    quad* light = new quad(point3(213, 554, 227), vec3(130, 0, 0), vec3(0, 0, 105), d_materials[3]);
    draw_list[obj_idx++] = light;
    d_lights[light_idx++] = light;

    // Two Hero Spheres (Glass vs Mirror)
    // Left: Glass sphere with refraction & internal reflection
    draw_list[obj_idx++] = new sphere(point3(190.0f, 90.0f, 190.0f), 90.0f, d_materials[4]);
    // Right: Reflective polished metal sphere
    draw_list[obj_idx++] = new sphere(point3(380.0f, 90.0f, 370.0f), 90.0f, d_materials[5]);

    for (int i = 0; i < obj_idx; ++i) {
        d_boxes[i] = draw_list[i]->bounding_box();
    }

    *d_obj_count = obj_idx;
    *d_proto_count = proto_idx;
    *d_light_count = light_idx;
}

constexpr int MAX_OBJECTS = 2000;
constexpr int MAX_MATERIALS = 2000;
constexpr int MAX_TEXTURES = 2000;
constexpr int MAX_PROTOTYPES = 256;
constexpr int MAX_LIGHTS = 200;
constexpr size_t DEVICE_HEAP_BYTES = 64ull * 1024ull * 1024ull;
constexpr int SAMPLES_PER_PASS = 8;


__global__ void create_primitives_kernel(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    hittable** d_lights,
    aabb* d_boxes,
    int* d_obj_count,
    int* d_proto_count,
    int* d_light_count,
    int kernel_selection,
    const ImageMetadata* d_scene_images
) {
    switch (kernel_selection) {
        case 1:
            create_snowman_kernel(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images
            );
            break;

        case 2:
            create_light_box(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images
            );
            break;

        case 3:
            create_quads(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images
            );
            break;

        case 4:
            create_earth_scene(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images
            );
            break;

        case 5:
            create_table_earth_scene(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images
            );
            break;
        case 6:
            create_spheres_benchmark_kernel(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images                
            );
        case 7:
            create_cornell_glass_kernel(
                d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
                d_boxes, d_obj_count, d_proto_count, d_light_count, d_scene_images                
            );
            break;
    }
}


__global__ void create_bvh_world_kernel(
    const FlatBVHNode* d_nodes,
    hittable** d_raw_list,
    const int* d_ordering,
    hittable** d_ordered_prims,
    flat_bvh** d_world_ptr,
    int num_objects
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (int i = 0; i < num_objects; ++i) {
        d_ordered_prims[i] = d_raw_list[d_ordering[i]];
    }

    *d_world_ptr = new flat_bvh(d_nodes, d_ordered_prims);
}


__global__ void free_world_kernel(
    hittable** d_raw_list,
    material** d_materials,
    texture** d_textures,
    hittable** d_prototypes,
    flat_bvh* d_world,
    int num_objects,
    int num_prototypes,
    int max_materials,
    int max_textures
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    if (d_world != nullptr) {
        delete d_world;
    }

    for (int i = 0; i < num_objects; ++i) {
        if (d_raw_list[i] != nullptr) {
            delete d_raw_list[i];
        }
    }

    for (int i = 0; i < num_prototypes; ++i) {
        if (d_prototypes[i] != nullptr) {
            delete d_prototypes[i];
        }
    }

    for (int i = 0; i < max_materials; ++i) {
        if (d_materials[i] != nullptr) {
            delete d_materials[i];
        }
    }

    for (int i = 0; i < max_textures; ++i) {
        if (d_textures[i] != nullptr) {
            delete d_textures[i];
        }
    }
}


// ----------------------------------------------------------------------------
// OPTIMIZED RENDER KERNEL
// ----------------------------------------------------------------------------
__global__ void render_subpass_kernel(
    colour* accum_fb,
    camera cam,
    flat_bvh* d_world,
    int pass_offset,
    int samples_to_render,
    hittable** lights,
    int num_lights
) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= cam.image_width || j >= cam.image_height) {
        return;
    }

    const int pixel_index = j * cam.image_width + i;

    unsigned int rng_state = pcg_hash(
        static_cast<unsigned int>(pixel_index)
        ^ pcg_hash(static_cast<unsigned int>(pass_offset))
    );

    colour subpass_color(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < samples_to_render; ++s) {
        ray r = cam.get_ray(i, j, &rng_state);

        subpass_color += ray_colour(
            r,
            d_world,
            cam.max_depth,
            &rng_state,
            cam.background,
            lights,
            num_lights
        );
    }

    accum_fb[pixel_index] += subpass_color;
}


__global__ void resolve_framebuffer_kernel(
    colour* fb,
    const colour* accum_fb,
    int total_pixels,
    float inv_total_samples
) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= total_pixels) {
        return;
    }

    fb[idx] = accum_fb[idx] * inv_total_samples;
}




// ----------------------------------------------------------------------------
// SCENE CONFIGURATION STRUCT
// ----------------------------------------------------------------------------
struct SceneConfig {
    int id = 0;
    const char* name = "unnamed";

    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 0;
    int samples_per_pixel = 0;
    int max_depth = 0;
    colour background = colour(0.0f, 0.0f, 0.0f);
    point3 look_from = point3(0.0f, 0.0f, 0.0f);
    point3 look_at = point3(0.0f, 0.0f, -1.0f);
    vec3 vup = vec3(0.0f, 1.0f, 0.0f);
    float vfov = 40.0f;

    int primitive_kernel_id = 0;

    std::vector<ImageMetadata> loaded_images;
};

void release_scene_images(SceneConfig& cfg) {
    for (ImageMetadata& meta : cfg.loaded_images) {
        if (meta.pixels != nullptr) {
            checkCudaErrors(cudaFree(meta.pixels));
            meta.pixels = nullptr;
        }

        meta.width = 0;
        meta.height = 0;
        meta.bytes_per_pixel = 0;
    }

    cfg.loaded_images.clear();
}

void upload_required_image(
    SceneConfig& cfg,
    const char* filename
) {
    image source(filename);

    unsigned char* d_pixels = nullptr;
    int width = 0;
    int height = 0;

    if (!source.copy_to_device(&d_pixels, &width, &height)) {
        release_scene_images(cfg);

        std::cerr
            << "ERROR: Failed to load and upload required texture '"
            << filename << "'.\n";

        std::exit(EXIT_FAILURE);
    }

    cfg.loaded_images.push_back({
        d_pixels,
        width,
        height,
        3
    });
}



void setup_scene_snowman(SceneConfig& cfg) {
    cfg.id = 1;
    cfg.name = "snowman";
    cfg.aspect_ratio = 16.0f / 9.0f;
    cfg.image_width = 1600;
    cfg.samples_per_pixel = 500;
    cfg.max_depth = 50;
    cfg.background = colour(0.70f, 0.80f, 1.00f);
    cfg.look_from = point3(10.0f, 3.5f, 3.5f);
    cfg.look_at = point3(0.0f, 2.5f, 0.0f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 30.0f;
    cfg.primitive_kernel_id = 1;
}


void setup_scene_lightbox(SceneConfig& cfg) {
    cfg.id = 2;
    cfg.name = "light_box";
    cfg.aspect_ratio = 16.0f / 9.0f;
    cfg.image_width = 1600;
    cfg.samples_per_pixel = 500;
    cfg.max_depth = 50;
    cfg.background = colour(0.0f, 0.0f, 0.0f);
    cfg.look_from = point3(10.0f, 3.5f, 3.5f);
    cfg.look_at = point3(0.0f, 2.5f, 0.0f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 30.0f;
    cfg.primitive_kernel_id = 2;
}


void setup_scene_quads(SceneConfig& cfg) {
    cfg.id = 3;
    cfg.name = "quads";
    cfg.aspect_ratio = 1.0f;
    cfg.image_width = 600;
    cfg.samples_per_pixel = 512;
    cfg.max_depth = 50;
    cfg.background = colour(0.0f, 0.0f, 0.0f);
    cfg.look_from = point3(278.0f, 278.0f, -800.0f);
    cfg.look_at = point3(278.0f, 278.0f, 0.0f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 40.0f;
    cfg.primitive_kernel_id = 3;
}


void setup_scene_earth(SceneConfig& cfg) {
    cfg.id = 4;
    cfg.name = "earth";
    cfg.aspect_ratio = 16.0f / 9.0f;
    cfg.image_width = 1200;
    cfg.samples_per_pixel = 500;
    cfg.max_depth = 50;
    cfg.background = colour(0.70f, 0.80f, 1.00f);
    cfg.look_from = point3(0.0f, 0.0f, 12.0f);
    cfg.look_at = point3(0.0f, 0.0f, 0.0f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 20.0f;
    cfg.primitive_kernel_id = 4;

    upload_required_image(cfg, "earthmap.jpg");
}


void setup_scene_table_earth(SceneConfig& cfg) {
    cfg.id = 5;
    cfg.name = "table_earth_room";
    cfg.aspect_ratio = 16.0f / 9.0f;
    cfg.image_width = 1200;
    cfg.samples_per_pixel = 200;
    cfg.max_depth = 50;
    cfg.background = colour(0.20f, 0.46f, 0.66f);

    // Position camera just off the table corner:
    // Close to corner (x=0.15, z=0.90) and slightly above (y=1.45)
    cfg.look_from = point3(-3.35f, 1.48f, 2.15f);
    cfg.look_at = point3(-1.90f, 1.22f, -1.15f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 38.0f;
    cfg.primitive_kernel_id = 5;

    // Load Earth texture to host, then transfer to GPU metadata
    upload_required_image(cfg, "earthmap.jpg");
    upload_required_image(cfg, "wooden.jpg");
}

void setup_scene_spheres_benchmark(SceneConfig& cfg) {
    cfg.id = 6;
    cfg.name = "spheres_1000_benchmark";
    cfg.aspect_ratio = 16.0f / 9.0f;
    cfg.image_width = 1600;
    cfg.samples_per_pixel = 256;
    cfg.max_depth = 50;
    cfg.background = colour(0.70f, 0.80f, 1.00f); // Clean sky blue
    cfg.look_from = point3(13.0f, 2.0f, 3.0f);
    cfg.look_at = point3(0.0f, 0.0f, 0.0f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 20.0f;
    cfg.primitive_kernel_id = 6;
}

// Stress Test: Dielectric/Glass Cornell Box (High warp divergence via Snell's Law & TIR)
void setup_scene_cornell_glass(SceneConfig& cfg) {
    cfg.id = 7;
    cfg.name = "cornell_glass_divergence";
    cfg.aspect_ratio = 1.0f;
    cfg.image_width = 600;
    cfg.samples_per_pixel = 512;
    cfg.max_depth = 50;
    cfg.background = colour(0.0f, 0.0f, 0.0f);
    cfg.look_from = point3(278.0f, 278.0f, -800.0f);
    cfg.look_at = point3(278.0f, 278.0f, 0.0f);
    cfg.vup = vec3(0.0f, 1.0f, 0.0f);
    cfg.vfov = 40.0f;
    cfg.primitive_kernel_id = 7;
}

// ----------------------------------------------------------------------------
// SHARED RENDER PIPELINE
// ----------------------------------------------------------------------------
void render_scene(SceneConfig& scene_cfg) {
    // Scene construction uses device-side new, so configure the CUDA device heap
    // before launching the primitive/world creation kernels.
    const auto total_start = std::chrono::steady_clock::now();
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_HEAP_BYTES));

    camera cam;
    cam.aspect_ratio = scene_cfg.aspect_ratio;
    cam.image_width = scene_cfg.image_width;
    cam.samples_per_pixel = scene_cfg.samples_per_pixel;
    cam.max_depth = scene_cfg.max_depth;
    cam.background = scene_cfg.background;
    cam.look_from = scene_cfg.look_from;
    cam.look_at = scene_cfg.look_at;
    cam.vup = scene_cfg.vup;
    cam.vfov = scene_cfg.vfov;

    cam.initialize();

    const int num_pixels = cam.image_width * cam.image_height;
    // Setups a frame buffer to store all pixels
    const size_t fb_size = static_cast<size_t>(num_pixels) * sizeof(colour);

    colour* d_fb = nullptr;
    colour* d_accum_fb = nullptr;

    checkCudaErrors(cudaMalloc(&d_fb, fb_size));
    checkCudaErrors(cudaMalloc(&d_accum_fb, fb_size));
    checkCudaErrors(cudaMemset(d_accum_fb, 0, fb_size));

    const int max_objects = MAX_OBJECTS;
    const int max_materials = MAX_MATERIALS;
    const int max_textures = MAX_TEXTURES;
    const int max_prototypes = MAX_PROTOTYPES;
    const int max_lights = MAX_LIGHTS;

    hittable** d_raw_list = nullptr;
    material** d_materials = nullptr;
    texture** d_textures = nullptr;
    hittable** d_prototypes = nullptr;
    hittable** d_lights = nullptr;
    aabb* d_boxes = nullptr;
    int* d_obj_count = nullptr;
    int* d_proto_count = nullptr;
    int* d_light_count = nullptr;
    hittable** d_ordered_prims = nullptr;
    flat_bvh** d_world_ptr = nullptr;

    checkCudaErrors(cudaMalloc(&d_raw_list, max_objects * sizeof(hittable*)));
    checkCudaErrors(cudaMalloc(&d_materials, max_materials * sizeof(material*)));
    checkCudaErrors(cudaMalloc(&d_textures, max_textures * sizeof(texture*)));
    checkCudaErrors(cudaMalloc(&d_prototypes, max_prototypes * sizeof(hittable*)));
    checkCudaErrors(cudaMalloc(&d_boxes, max_objects * sizeof(aabb)));
    checkCudaErrors(cudaMalloc(&d_lights, max_lights * sizeof(hittable*)));
    checkCudaErrors(cudaMalloc(&d_obj_count, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_proto_count, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_light_count, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_ordered_prims, max_objects * sizeof(hittable*)));
    checkCudaErrors(cudaMalloc(&d_world_ptr, sizeof(flat_bvh*)));

    // Initialise all pointer arrays and counters. This makes scene creation
    // and cleanup safe even if a scene has fewer than the maximum capacities.
    checkCudaErrors(cudaMemset(d_raw_list, 0, max_objects * sizeof(hittable*)));
    checkCudaErrors(cudaMemset(d_materials, 0, max_materials * sizeof(material*)));
    checkCudaErrors(cudaMemset(d_textures, 0, max_textures * sizeof(texture*)));
    checkCudaErrors(cudaMemset(d_prototypes, 0, max_prototypes * sizeof(hittable*)));
    checkCudaErrors(cudaMemset(d_lights, 0, max_lights * sizeof(hittable*)));
    checkCudaErrors(cudaMemset(d_obj_count, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(d_proto_count, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(d_light_count, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(d_world_ptr, 0, sizeof(flat_bvh*)));

    ImageMetadata* d_scene_images = nullptr;

    if (!scene_cfg.loaded_images.empty()) {
        const size_t meta_size =
            scene_cfg.loaded_images.size() * sizeof(ImageMetadata);

        checkCudaErrors(cudaMalloc(&d_scene_images, meta_size));
        checkCudaErrors(cudaMemcpy(d_scene_images, scene_cfg.loaded_images.data(),
                                   meta_size, cudaMemcpyHostToDevice));
    }

    std::cerr << "Creating primitives on GPU (scene: " << scene_cfg.name << ")...\n";

    create_primitives_kernel<<<1, 1>>>(
        d_raw_list, d_materials, d_textures, d_prototypes, d_lights,
        d_boxes, d_obj_count, d_proto_count, d_light_count,
        scene_cfg.primitive_kernel_id, d_scene_images);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int num_objects = 0;
    int num_prototypes = 0;
    int num_lights = 0;

    checkCudaErrors(cudaMemcpy(&num_objects, d_obj_count, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&num_prototypes, d_proto_count, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&num_lights, d_light_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (num_objects <= 0 || num_objects > max_objects) {
        std::cerr << "ERROR: Invalid object count returned by scene construction: "
                  << num_objects << "\n";
        std::exit(EXIT_FAILURE);
    }
    if (num_prototypes < 0 || num_prototypes > max_prototypes) {
        std::cerr << "ERROR: Invalid prototype count returned by scene construction: "
                  << num_prototypes << "\n";
        std::exit(EXIT_FAILURE);
    }
    if (num_lights < 0 || num_lights > max_lights) {
        std::cerr << "ERROR: Invalid light count returned by scene construction: "
                  << num_lights << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::vector<aabb> host_boxes(num_objects);
    checkCudaErrors(cudaMemcpy(host_boxes.data(), d_boxes,
                               static_cast<size_t>(num_objects) * sizeof(aabb),
                               cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // BUILD LINEAR BVH ON HOST (TIMED)
    // ------------------------------------------------------------------------
    std::cerr << "Building Linear BVH on Host with " << num_objects << " objects...\n";

    const auto bvh_start = std::chrono::steady_clock::now();

    std::vector<hittable*> host_proxies(num_objects);
    for (int i = 0; i < num_objects; ++i) {
        host_proxies[i] = new HostProxyPrimitive(i, host_boxes[i]);
    }

    std::vector<hittable*> ordered_proxies;
    std::vector<FlatBVHNode> flat_nodes;
    auto root = BVHBuilder::build(host_proxies, 0, num_objects, ordered_proxies);
    BVHBuilder::flatten(root, flat_nodes);

    const auto bvh_end = std::chrono::steady_clock::now();
    const double bvh_ms = std::chrono::duration<double, std::milli>(bvh_end - bvh_start).count();

    std::vector<int> ordered_indices(num_objects);
    for (int i = 0; i < num_objects; ++i) {
        ordered_indices[i] = static_cast<HostProxyPrimitive*>(ordered_proxies[i])->id;
        delete host_proxies[i];
    }

    std::cerr << "BVH complete: " << flat_nodes.size() << " flat nodes generated.\n";

    FlatBVHNode* d_flatnodes = nullptr;
    checkCudaErrors(cudaMalloc(&d_flatnodes, flat_nodes.size() * sizeof(FlatBVHNode)));
    checkCudaErrors(cudaMemcpy(d_flatnodes, flat_nodes.data(),
                               flat_nodes.size() * sizeof(FlatBVHNode),
                               cudaMemcpyHostToDevice));

    int* d_ordering = nullptr;
    checkCudaErrors(cudaMalloc(&d_ordering, ordered_indices.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_ordering, ordered_indices.data(),
                               ordered_indices.size() * sizeof(int),
                               cudaMemcpyHostToDevice));

    create_bvh_world_kernel<<<1, 1>>>(d_flatnodes, d_raw_list, d_ordering,
                                     d_ordered_prims, d_world_ptr, num_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    flat_bvh* d_world = nullptr;
    checkCudaErrors(cudaMemcpy(&d_world, d_world_ptr, sizeof(flat_bvh*),
                               cudaMemcpyDeviceToHost));
    if (d_world == nullptr) {
        std::cerr << "ERROR: Failed to construct GPU BVH world.\n";
        std::exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------
    // KERNEL LAUNCH CONFIGURATION
    // ------------------------------------------------------------------------
    const dim3 threads(16, 16);
    const dim3 blocks((cam.image_width + threads.x - 1) / threads.x,
                      (cam.image_height + threads.y - 1) / threads.y);

    const int resolve_threads = 256;
    const int resolve_blocks = (num_pixels + resolve_threads - 1) / resolve_threads;

    // ------------------------------------------------------------------------
    // WARM-UP PASS (1 sample, untimed - eliminates driver cold-start latency)
    // ------------------------------------------------------------------------
    render_subpass_kernel<<<blocks, threads>>>(
        d_accum_fb, cam, d_world, 0, 1, d_lights, num_lights);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemset(d_accum_fb, 0, fb_size));

    // ------------------------------------------------------------------------
    // GPU RENDERING PASSES (TIMED WITH CUDA EVENTS)
    // ------------------------------------------------------------------------
    std::cerr << "Rendering image with Linear BVH in multi-pass batches...\n";

    cudaEvent_t render_start;
    cudaEvent_t render_end;
    checkCudaErrors(cudaEventCreate(&render_start));
    checkCudaErrors(cudaEventCreate(&render_end));

    checkCudaErrors(cudaEventRecord(render_start));

    int remaining_samples = cam.samples_per_pixel;
    int pass_offset = 0;
    int pass_count = 0;

    while (remaining_samples > 0) {
        const int samples_this_pass = (remaining_samples > SAMPLES_PER_PASS)
                                          ? SAMPLES_PER_PASS
                                          : remaining_samples;

        render_subpass_kernel<<<blocks, threads>>>(
            d_accum_fb, cam, d_world, pass_offset, samples_this_pass, d_lights, num_lights);
        checkCudaErrors(cudaGetLastError());

        remaining_samples -= samples_this_pass;
        pass_offset += samples_this_pass;
        pass_count++;
    }

    // Resolve accumulator into final per-pixel colour
    resolve_framebuffer_kernel<<<resolve_blocks, resolve_threads>>>(
        d_fb, d_accum_fb, num_pixels, 1.0f / static_cast<float>(cam.samples_per_pixel));
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(render_end));
    checkCudaErrors(cudaEventSynchronize(render_end));

    float render_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&render_ms, render_start, render_end));

    // Compute Ray Tracing Throughput
    const double total_camera_samples = static_cast<double>(num_pixels) *
                                        static_cast<double>(cam.samples_per_pixel);
    const double camera_samples_per_second =
        total_camera_samples / (static_cast<double>(render_ms) / 1000.0);
    const double mrps = camera_samples_per_second / 1e6;

    std::cerr << "\n--- GPU Render Timing ---\n"
              << "Scene:                      " << scene_cfg.name << "\n"
              << "Resolution:                 " << cam.image_width << " x " << cam.image_height << "\n"
              << "Primitives:                 " << num_objects << "\n"
              << "Pixels:                     " << num_pixels << "\n"
              << "Samples per pixel:          " << cam.samples_per_pixel << "\n"
              << "Samples per pass:           " << SAMPLES_PER_PASS << "\n"
              << "Render passes:              " << pass_count << "\n"
              << "Host BVH Build time:        " << bvh_ms << " ms\n"
              << "GPU render time:            " << render_ms << " ms\n"
              << "Camera throughput:          " << mrps << " MRPS\n"
              << "-------------------------\n\n";

    checkCudaErrors(cudaEventDestroy(render_start));
    checkCudaErrors(cudaEventDestroy(render_end));

    // ------------------------------------------------------------------------
    // FRAMEBUFFER RETRIEVAL & FILE WRITE
    // ------------------------------------------------------------------------
    std::vector<colour> h_fb(num_pixels);
    checkCudaErrors(cudaMemcpy(h_fb.data(), d_fb, fb_size, cudaMemcpyDeviceToHost));

    std::cout << "P3\n" << cam.image_width << ' ' << cam.image_height << "\n255\n";
    for (int j = 0; j < cam.image_height; ++j) {
        for (int i = 0; i < cam.image_width; ++i) {
            const int pixel_index = j * cam.image_width + i;
            write_colour(std::cout, h_fb[pixel_index]);
        }
    }

    // ------------------------------------------------------------------------
    // CLEANUP MEMORY IN DEPENDENCY ORDER
    // ------------------------------------------------------------------------
    std::cerr << "Cleaning up memory...\n";

    free_world_kernel<<<1, 1>>>(
        d_raw_list, d_materials, d_textures, d_prototypes,
        d_world, num_objects, num_prototypes, max_materials, max_textures);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    release_scene_images(scene_cfg);

    if (d_scene_images != nullptr) {
        checkCudaErrors(cudaFree(d_scene_images));
    }
    if (d_flatnodes != nullptr) {
        checkCudaErrors(cudaFree(d_flatnodes));
    }
    if (d_ordering != nullptr) {
        checkCudaErrors(cudaFree(d_ordering));
    }
    if (d_ordered_prims != nullptr) {
        checkCudaErrors(cudaFree(d_ordered_prims));
    }
    if (d_boxes != nullptr) {
        checkCudaErrors(cudaFree(d_boxes));
    }
    if (d_obj_count != nullptr) {
        checkCudaErrors(cudaFree(d_obj_count));
    }
    if (d_proto_count != nullptr) {
        checkCudaErrors(cudaFree(d_proto_count));
    }
    if (d_raw_list != nullptr) {
        checkCudaErrors(cudaFree(d_raw_list));
    }
    if (d_prototypes != nullptr) {
        checkCudaErrors(cudaFree(d_prototypes));
    }
    if (d_materials != nullptr) {
        checkCudaErrors(cudaFree(d_materials));
    }
    if (d_lights != nullptr) {
        checkCudaErrors(cudaFree(d_lights));
    }
    if (d_light_count != nullptr) {
        checkCudaErrors(cudaFree(d_light_count));
    }
    if (d_textures != nullptr) {
        checkCudaErrors(cudaFree(d_textures));
    }
    if (d_world_ptr != nullptr) {
        checkCudaErrors(cudaFree(d_world_ptr));
    }
    if (d_accum_fb != nullptr) {
        checkCudaErrors(cudaFree(d_accum_fb));
    }
    if (d_fb != nullptr) {
        checkCudaErrors(cudaFree(d_fb));
    }

    const auto total_end = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(total_end - total_start).count();

    std::cerr << "Total pipeline time: " << total_seconds << " s\n";
    std::cerr << "Done.\n";
}


int main(int argc, char* argv[]) {
    int scene_id = 5; // default scene

    if (argc == 2) {
        try {
            scene_id = std::stoi(argv[1]);
        } catch (const std::exception&) {
            std::cerr << "Error: scene_id must be a valid integer.\n";
            std::cerr << "Usage: " << argv[0] << " [scene_id]\n";
            return EXIT_FAILURE;
        }
    } else if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [scene_id]\n";
        return EXIT_FAILURE;
    }

    if (scene_id < 1 || scene_id > 7) {
        std::cerr << "Error: scene_id must be between 1 and 5 inclusive.\n";
        return EXIT_FAILURE;
    }
    SceneConfig cfg;

    switch (scene_id) {
        case 1:
            setup_scene_snowman(cfg);
            break;

        case 2:
            setup_scene_lightbox(cfg);
            break;

        case 3:
            setup_scene_quads(cfg);
            break;

        case 4:
            setup_scene_earth(cfg);
            break;

        case 5:
            setup_scene_table_earth(cfg);
            break;
        case 6:
            setup_scene_spheres_benchmark(cfg);
            break;
        case 7:
            setup_scene_cornell_glass(cfg);
            break;
        default:
            std::cerr
                << "Unknown scene_id: "
                << scene_id
                << "\n";

            return EXIT_FAILURE;
    }

    render_scene(cfg);

    return EXIT_SUCCESS;
}
