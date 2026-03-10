#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include "CArray.cuh"
#include "CudaD3D12Texture.cuh"
#include "CoarseArray.cuh"
#include "Texturepack.cuh"
#include "renderer/Renderer.hpp" // Inherit from the new interface

// This is the concrete implementation of the Renderer for CUDA/D3D12.
class CudaRenderer : public Renderer {
public:
    CudaRenderer();
    ~CudaRenderer();

    // The implementation of the virtual Draw function
    void Draw(const Character& character, unsigned int frameCount) override;

    // All the CUDA/D3D12 specific data remains here
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

private:
    void drawCUDA(const glm::vec3& pos,
                  const glm::vec3& fo,
                  const glm::vec3& up,
                  const glm::vec3& ri,
                  const glm::mat4& unjitteredViewProjectionMatrix,
                  const glm::mat4& prevUnjitteredViewProjectionMatrix,
                  float jitterX, float jitterY);

    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;
};