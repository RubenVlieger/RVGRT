#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include "CArray.cuh"
#include "Framebuffer.cuh"
#include "CoarseArray.cuh"
#include "CuTex.cuh"

#include "Texturepack.cuh"

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
    Texturepack texturepack;
    CuTex shadowTex;
    CoarseArray csdf;
    CoarseArray GIdata;

    void drawCUDA(const glm::vec3& pos,
                  const glm::vec3& fo,
                  const glm::vec3& up,
                  const glm::vec3& ri);

private:
    // disallow copy
    StateRender(const StateRender&) = delete;
    StateRender& operator=(const StateRender&) = delete;
};
