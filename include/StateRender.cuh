#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include "CArray.cuh"
#include "Framebuffer.cuh"
#include "CSDF.cuh"


// A lightweight GPU-side RGBA pixel
// struct uchar4 {
//     unsigned char x, y, z, w;
//     __host__ __device__ uchar4() : x(0), y(0), z(0), w(255) {}
//     __host__ __device__ uchar4(unsigned char _x, unsigned char _y, unsigned char _z, unsigned char _w)
//         : x(_x), y(_y), z(_z), w(_w) {}
// };

// Main rendering class (host interface)
class StateRender {
public:
    StateRender();
    ~StateRender();

    // CPU-side voxel data (bit array)
    uint32_t* bitsArray;
    CArray cArray;
    Framebuffer framebuffer;
    CArray distBuffer;
    CSDF csdf;

    // GPU draw entrypoint
    //   pos = camera position
    //   fo  = camera forward vector
    //   up  = camera up vector
    //   ri  = camera right vector
    //   hostFrame = pointer to host framebuffer (output, must be pre-allocated)
    void drawCUDA(const glm::vec3& pos,
                  const glm::vec3& fo,
                  const glm::vec3& up,
                  const glm::vec3& ri);

private:
    // disallow copy
    StateRender(const StateRender&) = delete;
    StateRender& operator=(const StateRender&) = delete;
};
