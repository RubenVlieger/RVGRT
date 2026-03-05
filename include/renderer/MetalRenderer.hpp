#pragma once

#include "renderer/Renderer.hpp"
#include "renderer/Buffer.hpp" 
#include "renderer/MaterialMap.hpp"

#include "Texturepack.h"

#include <MetalFX/MetalFX.h>

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

    void Draw(id<MTLCommandBuffer> buffer, const Character& character, unsigned int frameCount);
    void Draw(const Character& character, unsigned int frameCount) override;

    id GetOutputTexture();

    void GenerateWorld();
    void OnResize(uint32_t newWidth, uint32_t newHeight);
    void generateNoiseTexture();

    void ResetScaler(); // Helper to reset history (e.g. on teleport)


    id GetCounterBuffer() { return _counterSampleBuffer; }
    id GetTimestampBuffer() { return _timestampBuffer; }

private:
    void createRenderTarget(uint32_t width, uint32_t height);
    Texturepack _texturepack;

    id _device;

    // --- Pipeline State Objects ---

    id _psoDistApprox;      // Kernel 0
    id _psoGBuffer;         // Kernel 1
    id _psoIndirect;        // Kernel 2
    id _psoAccumulate;      // Kernel 3
    id _psoDenoise;         // Kernel 4
    id _psoComposite;       // Kernel 5
    id _psoVolumetric; //
    id _psoExposure; 

    // screen textures
    id _texDirectLight;
    id _texAlbedo;
    id _texNormal;
    id _texMotion;
    id _texRawIndirect;
    id _texDenoised;
    id _texFinal;           // The main render target
    id _texFinalHistory[2];
    id _texDenoiseTemp;
    id _exposureBuffer;


    id _texVolumetric[2]; 

    id<MTLFXTemporalScaler> _temporalScaler;
    bool _scalerNeedsReset = true;
    id _texCompositeResult; 

    id _texDepth[2];        // [0] = current, [1] = prev (swaps every frame)
    id _texAccum[2];        // [0] = current, [1] = history (swaps every frame)
    
    id _halfDistTexture;

    MaterialMap _materialMap;

    //remaining data structures
    id _noiseTexture;

    //remaining data
    uint32_t _frameIndex = 0;
    id _counterSampleBuffer;
    id _timestampBuffer; 
    bool _supportsTimestamps;
};