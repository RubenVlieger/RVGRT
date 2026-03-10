#pragma once

#ifdef _WIN32
#include <cstdint>
#include <memory>
#include <glm/glm.hpp>
#include "renderer/Renderer.hpp"
#include "Texturepack.h"
#include "renderer/MaterialMap.hpp"
#include "CudaD3D12Texture.cuh"

struct CharacterGPUData;
class Character;

class CudaRenderer : public Renderer {
public:
    CudaRenderer();
    ~CudaRenderer() override;

    void Draw(const Character& character, unsigned int frameCount) override;
    void* GetOutputTexture() override;

    // Output target for D3D12 swapchain blit
    CudaD3D12Texture* GetOutputTexture() { return finalHistoryTex[frameIndex % 2].get(); }

private:
    void createRenderTargets(uint32_t width, uint32_t height);
    void DestroyRenderTargets();

    Texturepack texturepack;
    MaterialMap materialMap;

    uint32_t frameIndex = 0;

    // --- Intermediate Render Targets ---
    std::unique_ptr<CudaD3D12Texture> directLightTex;
    std::unique_ptr<CudaD3D12Texture> albedoTex;
    std::unique_ptr<CudaD3D12Texture> normalTex;
    std::unique_ptr<CudaD3D12Texture> motionTex;
    std::unique_ptr<CudaD3D12Texture> rawIndirectTex;
    std::unique_ptr<CudaD3D12Texture> denoisedTex;
    std::unique_ptr<CudaD3D12Texture> finalTex;
    std::unique_ptr<CudaD3D12Texture> denoiseTempTex;
    std::unique_ptr<CudaD3D12Texture> compositeResultTex;
    std::unique_ptr<CudaD3D12Texture> halfDistTex;
    
    // Ping-pong targets
    std::unique_ptr<CudaD3D12Texture> finalHistoryTex[2];
    std::unique_ptr<CudaD3D12Texture> volumetricTex[2];
    std::unique_ptr<CudaD3D12Texture> depthTex[2];
    std::unique_ptr<CudaD3D12Texture> accumTex[2];

    // GPU buffers for constants
    void* exposureBuffer = nullptr;
    void* characterBuffer = nullptr;

    // CUDA stream
    void* commandQueue = nullptr;

    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;
};
#endif