#pragma once

#include "renderer/Renderer.hpp"
#include "renderer/Buffer.hpp" 
#include "CArray.h"
#include "Texturepack.h"
#include "CoarseArray.h"

#include <cstdint> 
#include <memory> 

#ifdef __OBJC__
@protocol MTLComputePipelineState;
@protocol MTLTexture;
@protocol MTLDevice;
@protocol MTLCommandQueue;
#else
typedef void* id;
#endif

class Character; 

class MetalRenderer : public Renderer
{
public:
    MetalRenderer(id device);
    ~MetalRenderer() override;

    void Draw(id<MTLComputeCommandEncoder> encoder, const Character& character, unsigned int frameCount);
    void Draw(const Character& character, unsigned int frameCount) override;

    id GetOutputTexture();
    void GenerateWorld();
    void OnResize(uint32_t newWidth, uint32_t newHeight);
    void generateNoiseTexture();

    id GetCounterBuffer() { return _counterSampleBuffer; }
    id GetTimestampBuffer() { return _timestampBuffer; }

private:
    void createRenderTarget(uint32_t width, uint32_t height);
    Texturepack _texturepack;

    id _device;
    // --- Pipeline State Objects ---
    id _worldGenerationPSO; // For voxel generation
    id _distApproxPSO;      // Pre-pass (Accelerator)
    id _tiledDeferredPSO;   // Main TBDR Pass (The big new one)

    id _noiseTexture;

    id _renderTargetTexture; 
    id _halfDistTexture;

    id _voxelTexture; 

    id _counterSampleBuffer; // MTLCounterSampleBuffer (Opaque GPU storage)
    id _timestampBuffer;     // MTLBuffer (CPU readable storage)
    bool _supportsTimestamps;

    CoarseArray _csdf;
    CoarseArray _giData;
};