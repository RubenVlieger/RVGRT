#pragma once

#include "renderer/Renderer.hpp"
#include "renderer/CudaMaterialMap.cuh"
#include "Texturepack.h"
#include <cuda_runtime.h>

class Character;

/**
 * CudaRenderer - CUDA implementation of the 8-pass deferred pipeline.
 * 
 * Mirrors MetalRenderer exactly for maintainability and feature parity.
 * Uses CUDA Runtime API with the same pass order and buffer management.
 */
class CudaRenderer : public Renderer {
public:
    CudaRenderer();
    ~CudaRenderer() override;

    void Draw(const Character& character, unsigned int frameCount) override;
    
    // Matching MetalRenderer interface
    void GenerateWorld();
    void OnResize(uint32_t newWidth, uint32_t newHeight);
    void ResetScaler();
    
    // For D3D12 interop - get final output surface
    cudaSurfaceObject_t GetOutputSurface() const { return _texCompositeResult.surface; }
    cudaTextureObject_t GetOutputTexture() const { return _texCompositeResult.texture; }

private:
    void createRenderTarget(uint32_t width, uint32_t height);
    void freeRenderTargets();
    
    // Helper struct bundling array + surface + texture (mirrors MTLTexture concept)
    struct CudaRenderTarget {
        cudaArray_t array = nullptr;
        cudaSurfaceObject_t surface = 0;
        cudaTextureObject_t texture = 0;
        
        bool isValid() const { return array != nullptr; }
    };
    
    void allocateTarget(CudaRenderTarget& target, uint32_t width, uint32_t height, 
                        cudaChannelFormatDesc format);
    void freeTarget(CudaRenderTarget& target);
    
    // Same member name as MetalRenderer
    Texturepack _texturepack;
    
    // --- Render Targets (mirroring MetalRenderer.hpp lines 64-86) ---
    CudaRenderTarget _texDirectLight;      // RGBA16F
    CudaRenderTarget _texAlbedo;           // RGBA8
    CudaRenderTarget _texNormal;           // RGBA16F (was RGBA8Snorm in Metal, using 16F for simplicity)
    CudaRenderTarget _texMotion;           // RG16F
    CudaRenderTarget _texRawIndirect;      // RGBA16F
    CudaRenderTarget _texDenoised;         // RGBA16F
    CudaRenderTarget _texFinal;            // RGBA8
    CudaRenderTarget _texDenoiseTemp;      // RGBA16F
    CudaRenderTarget _texCompositeResult;  // RGBA16F
    
    CudaRenderTarget _texDepth[2];         // R32F, ping-pong
    CudaRenderTarget _texAccum[2];         // RGBA16F, ping-pong
    CudaRenderTarget _texFinalHistory[2];  // RGBA16F
    CudaRenderTarget _texVolumetric[2];    // RGBA16F, half-res, ping-pong
    
    CudaRenderTarget _halfDistTexture;     // R32F, half-res
    
    // --- Buffers (mirroring MetalRenderer) ---
    void* _exposureBuffer = nullptr;       // Device pointer to ExposureData
    void* _characterBuffer = nullptr;      // Device pointer to CharacterGPUData
    
    // --- Core Systems ---
    CudaMaterialMap _materialMap;
    
    // --- State (mirroring MetalRenderer.hpp lines 94-97) ---
    uint32_t _frameIndex = 0;
    uint32_t _width = 0;
    uint32_t _height = 0;
    bool _scalerNeedsReset = true;
    
    // CUDA stream for all rendering operations (like MTLCommandQueue)
    cudaStream_t _cudaStream = nullptr;
};
