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
    // Uses slab method
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        float inv_dir_x = 1.0f / r.direction().x();
        float inv_dir_y = 1.0f / r.direction().y();
        float inv_dir_z = 1.0f / r.direction().z();       
        int stack[16];
        int stack_ptr = 0;

        stack[stack_ptr++] = 0;

        bool hit_anything = false;
        float closest = ray_t.max;

        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr]; // pop next index of the stack
            const FlatBVHNode& node = d_nodes[node_idx];

            if (!node.bbox.hit(r, interval(ray_t.min, closest), inv_dir_x, inv_dir_y, inv_dir_z)) {
                // If this nodex bbox wasn't hit go to the next
                continue;
            }

            if (node.num_primitives > 0) {
                // Leaf node, check if primitives get hit

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
            } else {
                // Interior Node
                // If ray is in negative direction then the right object gets hit first so check that first.
        
                bool dir_is_neg = (r.direction()[node.axis] < 0.0f);
                if (dir_is_neg) {
                    stack[stack_ptr++] = node_idx + 1;             // near is right child
                    stack[stack_ptr++] = node.second_child_offset; // pop right first
                } else {
                    stack[stack_ptr++] = node.second_child_offset;
                    stack[stack_ptr++] = node_idx + 1;             // pop left first
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
        int stack[16];
        int stack_ptr = 0;

        stack[stack_ptr++] = 0;

        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr]; // pop next index of the stack
            const FlatBVHNode& node = d_nodes[node_idx];

            if (!node.bbox.hit(r, ray_t, inv_dir_x, inv_dir_y, inv_dir_z)) {
                // If this nodex bbox wasn't hit go to the next
                continue;
            }

            if (node.num_primitives > 0) {
                // Leaf node, check if primitives get hit
                hit_record temp_rec;
                for (int i = 0; i < node.num_primitives; i++) {
                    int prim_idx = node.primitive_offset + i;
                    if (d_primitives[prim_idx]->hit(r, ray_t, temp_rec)) {
                        return true;
                    }
                }
            } else {
                // Interior Node
                // If ray is in negative direction then the right object gets hit first so check that first.
        
                bool dir_is_neg = (r.direction()[node.axis] < 0.0f);
                if (dir_is_neg) {
                    stack[stack_ptr++] = node_idx + 1;             // near is right child
                    stack[stack_ptr++] = node.second_child_offset; // pop right first
                } else {
                    stack[stack_ptr++] = node.second_child_offset;
                    stack[stack_ptr++] = node_idx + 1;             // pop left first
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