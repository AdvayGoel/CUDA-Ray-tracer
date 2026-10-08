#ifndef BVH_CUH
#define BVH_CUH

#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cuda_runtime.h>

#include "aabb.cuh"
#include "hittable.h"

using std::shared_ptr;
using std::vector;

// Explicit stack to reduce memory spilling
struct RegisterStack8 {
    int s0, s1, s2, s3, s4, s5, s6, s7;
    int ptr = 0;

    __device__ __forceinline__ void push(int val) {
        if (ptr < 8) {
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
    }

    __device__ __forceinline__ int pop() {
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
    // Recursive algorithm
    // Takes all the 'primitives', the 'start' and 'end' index of all the primitives covered by this node, the current list of all the 'ordered_prims' that is being built
        static shared_ptr<BVHNode> build(vector<hittable*>& primitives, size_t start, size_t end, vector<hittable*>& ordered_prims) {
            auto node = std::make_shared<BVHNode>();
            size_t count = end - start;

            aabb bounds;
            for (size_t i = start; i < end; ++i) {
                bounds = (i == start) ? primitives[i]->bounding_box() : aabb(bounds, primitives[i]->bounding_box());
            }            
            node->bbox = bounds;

            if (count <= 2) {
                // If there are only 2 primitives covered by the node it becomes a leaf.
                node->first_prim_offset = static_cast<int>(ordered_prims.size());
                node->num_primitives = static_cast<int>(count);

                for (size_t i = start; i < end; ++i) {
                    ordered_prims.push_back(primitives[i]);
                }
                return node;                
            }
            
            // Split based on the axis with greatest variance
            // Makes bounding boxes less likely to overlap.
            int axis = 0;
            if (bounds.y.size() > bounds.x.size() && bounds.y.size() > bounds.z.size()) axis = 1;
            if (bounds.z.size() > bounds.x.size() && bounds.z.size() > bounds.x.size()) axis = 2;
            node->split_axis = axis;

            std::sort(primitives.begin() + start, primitives.begin() + end, [axis](const hittable* a, const hittable* b) {  return a->bounding_box().axis_interval(axis).min < b->bounding_box().axis_interval(axis).min; });

            size_t mid = start + count / 2;
            node->left = build(primitives, start, mid, ordered_prims);
            node->right = build(primitives, mid, end, ordered_prims);

            return node;
        }

        static int flatten(const shared_ptr<BVHNode>& node, vector<FlatBVHNode>& flat_nodes) {
            // Reserve slot for the parent node
            int my_index = static_cast<int>(flat_nodes.size());
            flat_nodes.emplace_back();

            FlatBVHNode flat;
            flat.bbox = node->bbox;
            flat.axis = static_cast<uint8_t>(node->split_axis);
            flat.num_primitives = static_cast<uint16_t>(node->num_primitives);
            flat.padding = 0;

            if (node->num_primitives > 0) {
                // Leaf node
                flat.primitive_offset = node->first_prim_offset;
            } else {
                // Interior
                flatten(node->left, flat_nodes);
                flat.second_child_offset = flatten(node->right, flat_nodes);
            }

            flat_nodes[my_index] = flat;
            return my_index;
        }
};

class flat_bvh : public hittable {
public:
    const FlatBVHNode* d_nodes;       // Flat array of BVH nodes in device memory
    hittable* const*   d_primitives;  // Array of scene primitives
    aabb               root_bbox;     // Bounding box of the entire scene

    __device__ flat_bvh(const FlatBVHNode* nodes, hittable* const* prims) : d_nodes(nodes), d_primitives(prims) {
        root_bbox = d_nodes[0].bbox;
    }
    // Uses slab method and push suprresion to reduce stack size
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
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
                    hittable* primitive = d_primitives[prim_idx];
                    if (primitive->hit(r, interval(ray_t.min, closest), temp_rec)) {
                        hit_anything = true;
                        temp_rec.object = primitive;
                        closest = temp_rec.t;
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
                    hittable* primitive = d_primitives[prim_idx];
                    if (primitive->hit(r, interval(ray_t.min, closest), temp_rec)) {
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
    HD aabb bounding_box() const override {
        return root_bbox;
    }
};
#endif