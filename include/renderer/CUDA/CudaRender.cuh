#pragma once

#include "renderer/Renderer.hpp"
#include "renderer/CUDA/CudaMaterialMap.cuh"
#include "Texturepack.h"
#include <cuda_runtime.h>

#ifdef _WIN32
// Forward declaration for Windows-specific interop
class CudaD3D12Texture;
#endif

class Character;

/**
 * CudaRenderer - CUDA implementation of the 8-pass deferred pipeline.
 * 
 * Mirrors MetalRenderer exactly for maintainability and feature parity.
 * Uses CUDA Runtime API with the same pass order and buffer management.
 * 
 * Features:
 * - 8-pass deferred rendering pipeline
 * - DLSS support via NVIDIA Streamline SDK (optional)
 * - CUDA-D3D12 interop for Windows
 */
class CudaRenderer : public Renderer {
public:
    CudaRenderer();
    ~CudaRenderer() override;

    void Draw(const Character& character, unsigned int frameCount) override;
    
    // Matching MetalRenderer interface
    void GenerateWorld();
    void OnResize(uint32_t renderW, uint32_t renderH, uint32_t screenW, uint32_t screenH);
    void ResetScaler();
    
    // DLSS Support
    void InitializeDLSS(void* d3dDevice, uint32_t width, uint32_t height);
    void UpdateDLSSConstants(float jitterX, float jitterY, bool reset);
    bool IsDLSSAvailable() const { return _dlssAvailable; }
    
    // Post-draw copy to output (D3D12 interop or DLSS input)
    void PostDraw(cudaSurfaceObject_t outputSurface, 
                  uint32_t width, uint32_t height,
                  bool useDLSS = false);
    
    // Getters for interop
    cudaSurfaceObject_t GetCompositeSurface() const { return _texCompositeResult.surface; }
    cudaTextureObject_t GetCompositeTexture() const { return _texCompositeResult.texture; }
    cudaSurfaceObject_t GetFinalSurface() const { return _texFinal.surface; }
    cudaTextureObject_t GetFinalTexture() const { return _texFinal.texture; }

private:
    void createRenderTarget(uint32_t width, uint32_t height);
    void freeRenderTargets();
    
    // Helper struct bundling array + surface + texture
    struct CudaRenderTarget {
        cudaArray_t array = nullptr;
        cudaSurfaceObject_t surface = 0;
        cudaTextureObject_t texture = 0;
        
        bool isValid() const { return array != nullptr; }
    };
    
    void allocateTarget(CudaRenderTarget& target, uint32_t width, uint32_t height, 
                        cudaChannelFormatDesc format);
    void freeTarget(CudaRenderTarget& target);
    
    // Texturepack (mirrors MetalRenderer)
    Texturepack _texturepack;
    
    // --- Render Targets (mirroring MetalRenderer) ---
    CudaRenderTarget _texDirectLight;      // RGBA16F
    CudaRenderTarget _texAlbedo;           // RGBA8
    CudaRenderTarget _texNormal;           // RGBA16F
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
    
    // --- Buffers ---
    void* _exposureBuffer = nullptr;       // Device pointer to ExposureData
    void* _characterBuffer = nullptr;      // Device pointer to CharacterGPUData
    
    // --- Core Systems ---
    CudaMaterialMap _materialMap;
    
    // --- State ---
    uint32_t _frameIndex = 0;
    uint32_t _width = 0;
    uint32_t _height = 0;
    bool _scalerNeedsReset = true;
    bool _dlssAvailable = false;
    
    // DLSS state
    float _jitterX = 0.0f;
    float _jitterY = 0.0f;
    
    // CUDA stream for all rendering operations
    cudaStream_t _cudaStream = nullptr;
};
