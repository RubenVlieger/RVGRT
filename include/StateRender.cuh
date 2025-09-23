#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include "CArray.cuh"
#include "CudaD3D12Texture.cuh"
#include "CoarseArray.cuh"

#include "Texturepack.cuh"

// Main rendering class (host interface)
class StateRender {
public:
    StateRender();
    ~StateRender();

    // CPU-side voxel data (bit array)
    uint32_t* bitsArray;

    CArray cArray;


    Texturepack texturepack;

    CoarseArray csdf;
    CoarseArray GIdata;

    CudaD3D12Texture lowResColorBuffer; 
    CudaD3D12Texture upscaledColorBuffer;

    CudaD3D12Texture motionVectorTex;
    CudaD3D12Texture depthTex;     
    CudaD3D12Texture shadowTex;
    CudaD3D12Texture halfDistBuffer;

    void drawCUDA(const glm::vec3& pos,
                  const glm::vec3& fo,
                  const glm::vec3& up,
                  const glm::vec3& ri,
                  glm::mat4* unjitteredViewProjectionMatrix,
                  glm::mat4* prevUnjitteredViewProjectionMatrix,
                  float jitterX, float jitterY);

private:
    // disallow copy
    StateRender(const StateRender&) = delete;
    StateRender& operator=(const StateRender&) = delete;
};
