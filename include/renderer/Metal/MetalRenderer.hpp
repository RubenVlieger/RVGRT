#pragma once

#include "renderer/RendererBase.hpp"
#include "renderer/RendererTraits.hpp"
#include "renderer/MaterialMap.hpp"
#include "Texturepack.h"

#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandBuffer;
@protocol MTLFXTemporalScaler;
@protocol MTLCounterSampleBuffer;
#else
typedef void* id;
#endif

class Character;

/**
 * MetalRenderer - Metal-specific renderer implementation
 * 
 * Inherits from RendererBase which handles the common pipeline logic.
 * This class only implements Metal-specific operations:
 * - Kernel loading from .metallib
 * - MetalFX temporal upscaling
 * - Timestamp sampling (optional)
 */
class MetalRenderer : public RendererBase<RendererImpl::MetalRendererTraits> {
public:
    explicit MetalRenderer(Device device);
    ~MetalRenderer() override;

    // Override Draw to accept Metal command buffer
    void Draw(CommandBuffer buffer, const Character& character, unsigned int frameCount);
    
    // Required by base class
    void Draw(const Character& character, unsigned int frameCount) override;

    // Platform-specific interface implementations
    void CreatePipelineStates() override;
    void DestroyPipelineStates() override;
    MaterialMap& GetMaterialMap() { return _materialMap; }
    Texturepack& GetTexturePack() override { return _texturepack; }
    id GetTemporalScaler() const override { return _temporalScaler; }
    void CreateExposureBuffer() override;
    void CreateCharacterBuffer() override;
    void UploadConstantData(CommandBuffer cmdBuf, 
                           const CameraData& camera,
                           const FrameData& frame,
                           const CharacterGPUData& characters) override;
    
    // Additional Metal-specific methods
    void GenerateWorld();
    id GetOutputTexture();
    id GetCounterBuffer() { return _counterSampleBuffer; }
    id GetTimestampBuffer() { return _timestampBuffer; }
    bool SupportsTimestamps() const { return _supportsTimestamps; }

private:
    void CreateTemporalScaler(uint32_t width, uint32_t height);
    void ClearHistoryBuffers();
    // Metal-specific members
    id _library;
    id _commandQueue;
    
    // Material map and texture pack
    MaterialMap _materialMap;
    Texturepack _texturepack;
    
    // MetalFX temporal scaler
    id _temporalScaler;
    
    // Optional timestamp support
    id _counterSampleBuffer;
    id _timestampBuffer;
    bool _supportsTimestamps;
    
    // Helper methods
    void SetupTimestampSupport();
};
