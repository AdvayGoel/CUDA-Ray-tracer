#ifndef BVH_CUH
#define BVH_CUH

#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "aabb.cuh"
#include "mat3.cuh"
#include "hittable.h"
#include <assert.h>

using std::shared_ptr;
using std::vector;



// Explicit stack to reduce memory spilling
struct RegisterStack8 {
    int s0, s1, s2, s3, s4, s5, s6, s7;
    int ptr = 0;

    __device__ __forceinline__ void push(int val) {
        assert(ptr >= 0 && ptr < 8);
        switch (ptr) {
            case 0: s0 = val; break;
            case 1: s1 = val; break;
            case 2: s2 = val; break;
            case 3: s3 = val; break;
            case 4: s4 = val; break;
            case 5: s5 = val; break;
            case 6: s6 = val; break;
            case 7: s7 = val; break;
        }
        ptr++;

    }

    __device__ __forceinline__ int pop() {
        assert(ptr > 0 && ptr <= 8);
        ptr--;
        switch (ptr) {
            case 0: return s0;
            case 1: return s1;
            case 2: return s2;
            case 3: return s3;
            case 4: return s4;
            case 5: return s5;
            case 6: return s6;
            case 7: return s7;
            default: return -1;
        }
    }

    __device__ __forceinline__ bool empty() const {
        return ptr <= 0;
    }
};

struct alignas(32) FlatBVHNode {
    // alligns each node in cache
    aabb bbox;

    // Each node is either a interior node (which bounds other nodes) or a leaf node (which represents a shape)
    // If it is a leaf then the union stored primitive_offset, which is the offset of the primitive in a list of ordered primitives
    // If it is a interior node then its first child is at index + 1 and second_child_offset holds the offset of the second child
    union {
        int primitive_offset;

        int second_child_offset;
    };

    // Stores how many primitives this Node bounds
    // 0 = interior node, 0< means leaf node
    uint16_t num_primitives;

    // 0 = x axis, 1 = y axis and 2 = z axis
    uint8_t axis;

    uint8_t padding;
};

struct BVHNode {
    aabb bbox;
    shared_ptr<BVHNode> left;
    shared_ptr<BVHNode> right;
    int first_prim_offset = 0;
    int num_primitives = 0;
    int split_axis = 0;
};

class BVHBuilder {
public:
    static constexpr int BIN_COUNT = 12;
    static constexpr size_t TARGET_LEAF_SIZE = 2;

    // Root has depth 0. A depth limit of 8 permits at most
    // eight deferred children in your current traversal.
    static constexpr int MAX_DEPTH = 8;

    static shared_ptr<BVHNode> build(
        vector<hittable*>& primitives,
        size_t start,
        size_t end,
        vector<hittable*>& ordered_prims
    ) {
        if (start >= end || end > primitives.size()) {
            throw std::invalid_argument(
                "BVHBuilder::build requires a valid, non-empty range"
            );
        }

        const size_t count = end - start;
        const size_t int_limit =
            static_cast<size_t>(std::numeric_limits<int>::max());

        if (ordered_prims.size() > int_limit ||
            count > int_limit - ordered_prims.size()) {
            throw std::overflow_error(
                "BVH primitive offsets exceed the int range"
            );
        }

        // Cache bounds and centroids once per build.
        // Recursive splitting then avoids repeatedly calling
        // bounding_box() through the CPU object interface.
        vector<BuildPrimitive> build_prims;
        build_prims.reserve(count);

        for (size_t i = start; i < end; ++i) {
            if (primitives[i] == nullptr) {
                throw std::invalid_argument(
                    "BVHBuilder received a null primitive"
                );
            }

            BuildPrimitive primitive;
            primitive.primitive = primitives[i];
            primitive.bounds = primitives[i]->bounding_box();

            for (int axis = 0; axis < 3; ++axis) {
                const double lo = axis_min(primitive.bounds, axis);
                const double hi = axis_max(primitive.bounds, axis);

                if (!std::isfinite(lo) ||
                    !std::isfinite(hi) ||
                    lo > hi) {
                    throw std::invalid_argument(
                        "BVHBuilder requires finite, valid bounds"
                    );
                }

                primitive.centroid[axis] = 0.5 * lo + 0.5 * hi;
            }

            build_prims.push_back(primitive);
        }

        auto root = build_recursive(
            build_prims,
            0,
            build_prims.size(),
            ordered_prims,
            0
        );

        // Preserve the old build() behaviour of reordering the
        // supplied primitive range as construction partitions it.
        for (size_t i = 0; i < count; ++i) {
            primitives[start + i] = build_prims[i].primitive;
        }

        return root;
    }

    static int flatten(
        const shared_ptr<BVHNode>& node,
        vector<FlatBVHNode>& flat_nodes
    ) {
        if (!node) {
            throw std::invalid_argument(
                "Cannot flatten a null BVH node"
            );
        }

        if (flat_nodes.size() >=
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "BVH node indices exceed the int range"
            );
        }

        const int my_index = static_cast<int>(flat_nodes.size());
        flat_nodes.emplace_back();

        FlatBVHNode flat{};
        flat.bbox = node->bbox;
        flat.axis = static_cast<uint8_t>(node->split_axis);
        flat.padding = 0;

        if (node->num_primitives > 0) {
            if (node->num_primitives >
                std::numeric_limits<uint16_t>::max()) {
                throw std::overflow_error(
                    "BVH leaf primitive count exceeds uint16_t"
                );
            }

            flat.num_primitives =
                static_cast<uint16_t>(node->num_primitives);

            flat.primitive_offset = node->first_prim_offset;
        } else {
            flat.num_primitives = 0;

            // Preorder flattening preserves left_child = index + 1.
            flatten(node->left, flat_nodes);
            flat.second_child_offset =
                flatten(node->right, flat_nodes);
        }

        flat_nodes[my_index] = flat;
        return my_index;
    }

    // Optional CPU-side diagnostic. Depth is counted in edges.
    static int maximum_depth(const shared_ptr<BVHNode>& node) {
        if (!node || node->num_primitives > 0) {
            return 0;
        }

        return 1 + std::max(
            maximum_depth(node->left),
            maximum_depth(node->right)
        );
    }

private:
    struct BuildPrimitive {
        hittable* primitive = nullptr;
        aabb bounds;
        double centroid[3]{};
    };

    struct Bin {
        aabb bounds;
        size_t count = 0;

        void add(const aabb& other_bounds, size_t other_count) {
            if (other_count == 0) {
                return;
            }

            // Do not assume the default aabb is an empty box.
            if (count == 0) {
                bounds = other_bounds;
            } else {
                bounds = aabb(bounds, other_bounds);
            }

            count += other_count;
        }
    };

    static double axis_min(const aabb& bounds, int axis) {
        if (axis == 0) return bounds.x.min;
        if (axis == 1) return bounds.y.min;
        return bounds.z.min;
    }

    static double axis_max(const aabb& bounds, int axis) {
        if (axis == 0) return bounds.x.max;
        if (axis == 1) return bounds.y.max;
        return bounds.z.max;
    }

    static double surface_area(const aabb& bounds) {
        const double dx = axis_max(bounds, 0) - axis_min(bounds, 0);
        const double dy = axis_max(bounds, 1) - axis_min(bounds, 1);
        const double dz = axis_max(bounds, 2) - axis_min(bounds, 2);

        return 2.0 * (dx * dy + dx * dz + dy * dz);
    }

    static int bin_index(
        double centroid,
        double centroid_min,
        double centroid_extent
    ) {
        const double position =
            (centroid - centroid_min) / centroid_extent;

        int index = static_cast<int>(position * BIN_COUNT);

        // The maximum centroid maps to BIN_COUNT before clamping.
        index = std::max(0, std::min(index, BIN_COUNT - 1));
        return index;
    }

    static void make_leaf(
        const shared_ptr<BVHNode>& node,
        const vector<BuildPrimitive>& primitives,
        size_t start,
        size_t end,
        vector<hittable*>& ordered_prims
    ) {
        const size_t count = end - start;

        if (count >
            static_cast<size_t>(
                std::numeric_limits<uint16_t>::max())) {
            throw std::overflow_error(
                "Depth-limited BVH leaf exceeds uint16_t capacity"
            );
        }

        node->first_prim_offset =
            static_cast<int>(ordered_prims.size());

        node->num_primitives = static_cast<int>(count);

        for (size_t i = start; i < end; ++i) {
            ordered_prims.push_back(primitives[i].primitive);
        }
    }

    static shared_ptr<BVHNode> build_recursive(
        vector<BuildPrimitive>& primitives,
        size_t start,
        size_t end,
        vector<hittable*>& ordered_prims,
        int depth
    ) {
        auto node = std::make_shared<BVHNode>();
        const size_t count = end - start;

        aabb bounds = primitives[start].bounds;

        double centroid_min[3];
        double centroid_max[3];

        for (int axis = 0; axis < 3; ++axis) {
            centroid_min[axis] = primitives[start].centroid[axis];
            centroid_max[axis] = primitives[start].centroid[axis];
        }

        for (size_t i = start + 1; i < end; ++i) {
            bounds = aabb(bounds, primitives[i].bounds);

            for (int axis = 0; axis < 3; ++axis) {
                centroid_min[axis] = std::min(
                    centroid_min[axis],
                    primitives[i].centroid[axis]
                );

                centroid_max[axis] = std::max(
                    centroid_max[axis],
                    primitives[i].centroid[axis]
                );
            }
        }

        node->bbox = bounds;

        if (count <= TARGET_LEAF_SIZE || depth >= MAX_DEPTH) {
            make_leaf(node, primitives, start, end, ordered_prims);
            return node;
        }

        int best_axis = -1;
        int best_boundary = -1;
        double best_cost = std::numeric_limits<double>::infinity();

        // Evaluate all 11 bin boundaries on all usable axes.
        for (int axis = 0; axis < 3; ++axis) {
            const double extent =
                centroid_max[axis] - centroid_min[axis];

            if (!(extent > 0.0)) {
                continue;
            }

            std::array<Bin, BIN_COUNT> bins;

            for (size_t i = start; i < end; ++i) {
                const int index = bin_index(
                    primitives[i].centroid[axis],
                    centroid_min[axis],
                    extent
                );

                bins[index].add(primitives[i].bounds, 1);
            }

            std::array<Bin, BIN_COUNT> prefix;
            std::array<Bin, BIN_COUNT> suffix;

            Bin running_left;

            for (int i = 0; i < BIN_COUNT; ++i) {
                running_left.add(bins[i].bounds, bins[i].count);

                if (running_left.count != 0) {
                    prefix[i].add(
                        running_left.bounds,
                        running_left.count
                    );
                }
            }

            Bin running_right;

            for (int i = BIN_COUNT - 1; i >= 0; --i) {
                running_right.add(bins[i].bounds, bins[i].count);

                if (running_right.count != 0) {
                    suffix[i].add(
                        running_right.bounds,
                        running_right.count
                    );
                }
            }

            for (int boundary = 0;
                 boundary < BIN_COUNT - 1;
                 ++boundary) {
                const Bin& left = prefix[boundary];
                const Bin& right = suffix[boundary + 1];

                // Never create an empty child.
                if (left.count == 0 || right.count == 0) {
                    continue;
                }

                const double cost =
                    surface_area(left.bounds)
                        * static_cast<double>(left.count)
                    + surface_area(right.bounds)
                        * static_cast<double>(right.count);

                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_boundary = boundary;
                }
            }
        }

        size_t mid = start;
        int split_axis = best_axis;

        if (best_axis >= 0) {
            const double minimum = centroid_min[best_axis];
            const double extent =
                centroid_max[best_axis] - minimum;

            auto middle = std::partition(
                primitives.begin() + start,
                primitives.begin() + end,
                [=](const BuildPrimitive& primitive) {
                    return bin_index(
                        primitive.centroid[best_axis],
                        minimum,
                        extent
                    ) <= best_boundary;
                }
            );

            mid = static_cast<size_t>(
                middle - primitives.begin()
            );
        }

        // Fallback for coincident centroids or an invalid partition.
        if (best_axis < 0 || mid == start || mid == end) {
            split_axis = 0;

            for (int axis = 1; axis < 3; ++axis) {
                const double extent =
                    centroid_max[axis] - centroid_min[axis];

                const double current_extent =
                    centroid_max[split_axis]
                    - centroid_min[split_axis];

                if (extent > current_extent) {
                    split_axis = axis;
                }
            }

            mid = start + count / 2;

            std::nth_element(
                primitives.begin() + start,
                primitives.begin() + mid,
                primitives.begin() + end,
                [split_axis](
                    const BuildPrimitive& a,
                    const BuildPrimitive& b
                ) {
                    return a.centroid[split_axis]
                         < b.centroid[split_axis];
                }
            );
        }

        node->split_axis = split_axis;
        node->num_primitives = 0;

        node->left = build_recursive(
            primitives, start, mid, ordered_prims, depth + 1
        );

        node->right = build_recursive(
            primitives, mid, end, ordered_prims, depth + 1
        );

        return node;
    }
};

class flat_bvh {
public:
    const FlatBVHNode* d_nodes;       // Flat array of BVH nodes in device memory
    const gpu_primitive *d_primitives;  // Array of scene primitives
    aabb               root_bbox;     // Bounding box of the entire scene

    __device__ flat_bvh(const FlatBVHNode* nodes, const gpu_primitive *prims) : d_nodes(nodes), d_primitives(prims) {
        root_bbox = d_nodes[0].bbox;
    }
    // Uses slab method and push suprresion to reduce stack size
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        float inv_dir_x = 1.0f / r.direction().x();
        float inv_dir_y = 1.0f / r.direction().y();
        float inv_dir_z = 1.0f / r.direction().z();

        RegisterStack8 stack;

        int node_idx = 0; // Start at root directly
        bool hit_anything = false;
        float closest = ray_t.max;

        // Test root bounding box once before starting
        if (!d_nodes[0].bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z)) {
            return false;
        }

        while (node_idx != -1) {
            const FlatBVHNode& node = d_nodes[node_idx];

            if (node.num_primitives > 0) {
                // Leaf node: intersect primitives
                for (int i = 0; i < node.num_primitives; i++) {
                    int prim_idx = node.primitive_offset + i;
                    hit_record temp_rec;
                    const gpu_primitive& primitive = d_primitives[prim_idx];
                    if (primitive.hit(r, interval(ray_t.min, closest), temp_rec)) {
                        hit_anything = true;
                        closest = temp_rec.t;
                        temp_rec.object = primitive.object;
                        rec = temp_rec;
                    }
                }
                // Pop the next node from the stack
                node_idx = !stack.empty() ? stack.pop() : -1;
            } else {
                // Interior node: determine child indices
                int left_child = node_idx + 1;
                int right_child = node.second_child_offset;

                // Determine near and far child based on ray direction
                bool dir_is_neg = (r.direction()[node.axis] < 0.0f);
                int near_child = dir_is_neg ? right_child : left_child;
                int far_child  = dir_is_neg ? left_child  : right_child;

                // Test intersection against both children
                bool hit_near = d_nodes[near_child].bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z);
                bool hit_far  = d_nodes[far_child].bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z);

                if (hit_near && hit_far) {
                    // Both hit: push the far child, traverse into the near child
                    stack.push(far_child);
                    node_idx = near_child;
                } else if (hit_near) {
                    // Only near child hit: descend without touching the stack
                    node_idx = near_child;
                } else if (hit_far) {
                    // Only far child hit: descend without touching the stack
                    node_idx = far_child;
                } else {
                    // Neither hit: pop from the stack
                    node_idx = !stack.empty() ? stack.pop() : -1;
                }
            }
        }

        return hit_anything;
    }
    // Checks if a shadow ray hits the scene
    __device__ bool occluded(const ray& r, interval ray_t) const {
        float inv_dir_x = 1.0f / r.direction().x();
        float inv_dir_y = 1.0f / r.direction().y();
        float inv_dir_z = 1.0f / r.direction().z();

        RegisterStack8 stack;

        int node_idx = 0; // Start at root directly
        float closest = ray_t.max;

        // Test root bounding box once before starting
        if (!d_nodes[0].bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z)) {
            return false;
        }

        while (node_idx != -1) {
            const FlatBVHNode& node = d_nodes[node_idx];

            if (node.num_primitives > 0) {
                // Leaf node: intersect primitives
                for (int i = 0; i < node.num_primitives; i++) {
                    int prim_idx = node.primitive_offset + i;
                    hit_record temp_rec;
                    const gpu_primitive& primitive = d_primitives[prim_idx];
                    if (primitive.hit(r, interval(ray_t.min, closest), temp_rec)) {
                        return true;
                    }
                }
                // Pop the next node from the stack
                node_idx = !stack.empty() ? stack.pop() : -1;
            } else {
                // Interior node: determine child indices
                int left_child = node_idx + 1;
                int right_child = node.second_child_offset;

                // Determine near and far child based on ray direction
                bool dir_is_neg = (r.direction()[node.axis] < 0.0f);
                int near_child = dir_is_neg ? right_child : left_child;
                int far_child  = dir_is_neg ? left_child  : right_child;

                // Test intersection against both children
                bool hit_near = d_nodes[near_child].bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z);
                bool hit_far  = d_nodes[far_child].bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z);

                if (hit_near && hit_far) {
                    // Both hit: push the far child, traverse into the near child
                    stack.push(far_child);
                    node_idx = near_child;
                } else if (hit_near) {
                    // Only near child hit: descend without touching the stack
                    node_idx = near_child;
                } else if (hit_far) {
                    // Only far child hit: descend without touching the stack
                    node_idx = far_child;
                } else {
                    // Neither hit: pop from the stack
                    node_idx = !stack.empty() ? stack.pop() : -1;
                }
            }
        }

        return false;
    }
    HD aabb bounding_box() const {
        return root_bbox;
    }
};
#endif