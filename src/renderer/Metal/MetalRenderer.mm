#include "renderer/Metal/MetalRenderer.hpp"
#include "Character.hpp"
#include "State.hpp"
#include "Texturepack.h"
#include "cumath.h"
#include "renderer/MaterialMap.hpp"
#include "renderer/ShaderTypes.h"
#include <MetalFX/MetalFX.h>
#import <MetalKit/MetalKit.h>
#include <cassert>

@protocol MTLFXTemporalScaler_Unlocked <NSObject>
@property(readwrite, nonatomic) simd_float2 motionVectorScale;
@property(readwrite, nonatomic) simd_float2 jitterOffset;
@end

MetalRenderer::MetalRenderer(Device device)
    : RendererBase<RendererImpl::MetalRendererTraits>(device)
    , _materialMap()
    , _texturepack()
    , _library(nullptr)
    , _commandQueue(nullptr)
    , _temporalScaler(nullptr)
    , _counterSampleBuffer(nullptr)
    , _timestampBuffer(nullptr)
    , _supportsTimestamps(false)
    , _psoTextOverlay(nullptr)
{
    // Create command queue
    _commandQueue = [device newCommandQueue];
    
    // Load Metal library
    NSError *error = nil;
    _library = [device newDefaultLibrary];
    if (!_library) {
        NSLog(@"FATAL: Could not load default.metallib.");
        abort();
    }
    
    // Create pipeline states
    CreatePipelineStates();
    
    // Create GPU buffers
    CreateExposureBuffer();
    CreateCharacterBuffer();
    
    // Initialize render targets
    Initialize(State::dispWIDTH, State::dispHEIGHT);
    
    // Create MetalFX temporal scaler
    CreateTemporalScaler(State::dispWIDTH, State::dispHEIGHT);
    
    // Clear history buffers to avoid uninitialized memory artifacts
    ClearHistoryBuffers();
    
    // Setup timestamp support (optional)
    SetupTimestampSupport();
    
    // Initialize text rendering
    id<MTLDevice> metalDevice = (id<MTLDevice>)_device;
    if (_fontAtlas.InitializeWithSystemFont(TEXT_FONT_SIZE)) {
        if (_textRenderer.Initialize(metalDevice, _fontAtlas)) {
            NSLog(@"[TextRenderer] Initialized successfully with SDF font atlas");
        } else {
            NSLog(@"[TextRenderer] WARNING: Failed to initialize renderer");
        }
    } else {
        NSLog(@"[TextRenderer] WARNING: Failed to load system font");
    }
    
    // Generate world
    NSLog(@"Starting Dynamic World Generation (XBrickMap)...");
    _materialMap.GenerateDynamic();
    NSLog(@"World Generation Complete.");
}

MetalRenderer::~MetalRenderer() {
    DestroyPipelineStates();
    Shutdown();
}

void MetalRenderer::CreatePipelineStates() {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    NSError *error = nil;
    
    _psoDistApprox = [device newComputePipelineStateWithFunction:
                       [_library newFunctionWithName:@"distApproximationKernel"]
                                                            error:&error];
    _psoGBuffer = [device newComputePipelineStateWithFunction:
                    [_library newFunctionWithName:@"GBufferAndDirectLight"]
                                                         error:&error];
    _psoIndirect = [device newComputePipelineStateWithFunction:
                     [_library newFunctionWithName:@"IndirectBounce"]
                                                          error:&error];
    _psoAccumulate = [device newComputePipelineStateWithFunction:
                       [_library newFunctionWithName:@"TemporalAccumulation"]
                                                            error:&error];
    _psoDenoise = [device newComputePipelineStateWithFunction:
                    [_library newFunctionWithName:@"BilateralDenoise"]
                                                         error:&error];
    _psoComposite = [device newComputePipelineStateWithFunction:
                      [_library newFunctionWithName:@"Composite"]
                                                           error:&error];
    _psoVolumetric = [device newComputePipelineStateWithFunction:
                       [_library newFunctionWithName:@"VolumetricFog"]
                                                            error:&error];
_psoExposure = [device newComputePipelineStateWithFunction:
                      [_library newFunctionWithName:@"ComputeExposure"]
                                                           error:&error];
    _psoTextOverlay = [device newComputePipelineStateWithFunction:
                         [_library newFunctionWithName:@"TextOverlay"]
                                                              error:&error];
    
    if (!_psoDistApprox || !_psoGBuffer || !_psoIndirect || !_psoAccumulate ||
        !_psoDenoise || !_psoComposite || !_psoVolumetric || !_psoExposure || !_psoTextOverlay) {
        NSLog(@"FATAL: Failed to load kernels. Error: %@", error);
        abort();
    }
}

void MetalRenderer::DestroyPipelineStates() {
    // ARC handles release automatically
    _psoDistApprox = nullptr;
    _psoGBuffer = nullptr;
    _psoIndirect = nullptr;
    _psoAccumulate = nullptr;
    _psoDenoise = nullptr;
    _psoComposite = nullptr;
    _psoVolumetric = nullptr;
    _psoExposure = nullptr;
    _psoTextOverlay = nullptr;
    _library = nullptr;
}

void MetalRenderer::CreateExposureBuffer() {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    ExposureData expData = {};
    expData.sceneLuminance = 0.5f;
    _exposureBuffer = [device newBufferWithBytes:&expData
                                           length:sizeof(ExposureData)
                                          options:MTLResourceStorageModeShared];
}

void MetalRenderer::CreateCharacterBuffer() {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    _characterBuffer = [device newBufferWithLength:sizeof(CharacterGPUData)
                                            options:MTLResourceStorageModeShared];
}

void MetalRenderer::SetupTimestampSupport() {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    
    _supportsTimestamps = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    
    if (_supportsTimestamps) {
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> set in [device counterSets]) {
            if ([set.name caseInsensitiveCompare:@"timestamp"] == NSOrderedSame) {
                timestampSet = set;
                break;
            }
        }
        
        if (timestampSet) {
            MTLCounterSampleBufferDescriptor *desc = [[MTLCounterSampleBufferDescriptor alloc] init];
            desc.counterSet = timestampSet;
            desc.label = @"TimestampCounter";
            desc.sampleCount = 18;
            desc.storageMode = MTLStorageModePrivate;
            _counterSampleBuffer = [device newCounterSampleBufferWithDescriptor:desc error:nil];
            _timestampBuffer = [device newBufferWithLength:18 * sizeof(uint64_t)
                                                    options:MTLResourceStorageModeShared];
        }
    }
}

void MetalRenderer::CreateTemporalScaler(uint32_t width, uint32_t height) {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    
    MTLFXTemporalScalerDescriptor *scalerDesc = [[MTLFXTemporalScalerDescriptor alloc] init];
    scalerDesc.inputWidth = width;
    scalerDesc.inputHeight = height;
    scalerDesc.outputWidth = width;
    scalerDesc.outputHeight = height;
    scalerDesc.colorTextureFormat = MTLPixelFormatRGBA16Float;
    scalerDesc.depthTextureFormat = MTLPixelFormatR32Float;
    scalerDesc.motionTextureFormat = MTLPixelFormatRG16Float;
    scalerDesc.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
    
    _temporalScaler = [scalerDesc newTemporalScalerWithDevice:device];
    _scalerNeedsReset = true;
    
    NSLog(@"MetalFX Temporal Scaler created: %dx%d -> %dx%d", width, height, width, height);
}

void MetalRenderer::ClearHistoryBuffers() {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    auto& manager = _renderTargetManager;
    
    // Clear textures using render pass with clear color
    MTLClearColor clearColor = MTLClearColorMake(0, 0, 0, 0);
    
    auto clearTexture = [&](id<MTLTexture> texture, MTLPixelFormat format) {
        MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
        passDesc.colorAttachments[0].texture = texture;
        passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
        passDesc.colorAttachments[0].clearColor = clearColor;
        
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
        renderEncoder.label = @"ClearTexture";
        [renderEncoder endEncoding];
    };
    
    // Clear all ping-pong buffers to black
    for (int i = 0; i < 2; i++) {
        clearTexture((id<MTLTexture>)manager.GetDepth(i).texture, MTLPixelFormatR32Float);
        clearTexture((id<MTLTexture>)manager.GetAccum(i).texture, MTLPixelFormatRGBA16Float);
        clearTexture((id<MTLTexture>)manager.GetFinalHistory(i).texture, MTLPixelFormatRGBA16Float);
        clearTexture((id<MTLTexture>)manager.GetVolumetric(i).texture, MTLPixelFormatRGBA16Float);
    }
    
    // Clear other targets
    clearTexture((id<MTLTexture>)manager.GetCompositeResult().texture, MTLPixelFormatRGBA16Float);
    clearTexture((id<MTLTexture>)manager.GetRawIndirect().texture, MTLPixelFormatRGBA16Float);
    clearTexture((id<MTLTexture>)manager.GetDenoised().texture, MTLPixelFormatRGBA16Float);
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    NSLog(@"History buffers cleared");
}

void MetalRenderer::UploadConstantData(CommandBuffer cmdBuf,
                                       const CameraData& camera,
                                       const FrameData& frame,
                                       const CharacterGPUData& characters) {
    (void)cmdBuf;
    
    // Upload character data to GPU buffer
    CharacterGPUData *charDataDest = (CharacterGPUData *)[(id<MTLBuffer>)_characterBuffer contents];
    memcpy(charDataDest, &characters, sizeof(CharacterGPUData));
    
    // Camera and frame data are passed via setBytes in the encoder
    // Store them temporarily to be used by ExecutePipeline
    // This is a bit of a hack - ideally we'd pass them through the encoder
    // But since ExecutePipeline creates the encoder, we need another approach
    
    // For now, we'll store them in member variables and override ExecutePipeline
    // Actually, better approach: the base class should pass these to the pass methods
    // Let's leave this as a TODO and use the old Draw method for now
}

void MetalRenderer::Draw(CommandBuffer cmdBuf,
                         const Character& character, unsigned int frameCount) {
    int currIdx = _frameIndex % 2;
    int prevIdx = (_frameIndex + 1) % 2;
    
    // Prepare data using FrameDataManager
    simd_int3 worldOrigin = _materialMap.GetWorldOrigin();
    CameraData camData = _frameDataManager.PrepareCameraData(character);
    FrameData frameData = _frameDataManager.PrepareFrameData(_frameIndex, worldOrigin);
    CharacterGPUData charData = _frameDataManager.PrepareCharacterData(
        character, State::state.otherCharacters);
    
    // Update material map streaming
    simd_float3 camPos = camData.position;
    bool sectorsChanged = _materialMap.UpdateStreaming(camPos);
    if (sectorsChanged) {
        _scalerNeedsReset = true;
    }
    
    // Prepare text overlay (Hello World placeholder)
    if (_fontAtlas.IsValid()) {
        _textRenderer.BeginFrame(State::dispWIDTH, State::dispHEIGHT);
        _textRenderer.AddText("Hello World", 20.0f, 20.0f, 1.0f,
                              simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f), 0.03f, false, 1e30f);
        _textRenderer.EndFrame();
        _textRenderer.UpdateBuffers((id<MTLDevice>)_device);
    }
    
    // Upload character data
    CharacterGPUData *charDataDest = (CharacterGPUData *)[(id<MTLBuffer>)_characterBuffer contents];
    memcpy(charDataDest, &charData, sizeof(CharacterGPUData));
    
    // Create compute encoder
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"Render Pipeline";
    
    // Grid sizes
    MTLSize gridSizeFull = MTLSizeMake(_width, _height, 1);
    MTLSize gridSizeHalf = MTLSizeMake(_width / 2, _height / 2, 1);
    MTLSize groupSize = MTLSizeMake(16, 16, 1);
    MTLSize groupSize8 = MTLSizeMake(8, 8, 1);
    
    // Get render target textures from manager
    auto& manager = _renderTargetManager;
    
    // Pass 0: Distance Approximation
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:0 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 0: Approx"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoDistApprox];
    [encoder setTexture:(id<MTLTexture>)manager.GetHalfDist().texture atIndex:0];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorMaskBuffer() offset:0 atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:7];
    [encoder dispatchThreads:gridSizeHalf threadsPerThreadgroup:groupSize8];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:1 withBarrier:YES];
    }
    
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    // Pass 1: GBuffer
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:2 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 1: GBuffer"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoGBuffer];
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetAlbedo().texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetNormal().texture atIndex:2];
    [encoder setTexture:(id<MTLTexture>)manager.GetMotion().texture atIndex:3];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:4];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorMaskBuffer() offset:0 atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:7];
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:8];
    [encoder setTexture:(id<MTLTexture>)manager.GetHalfDist().texture atIndex:9];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:3 withBarrier:YES];
    }
    
    // Pass 2: Indirect
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:4 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 2: Indirect"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoIndirect];
    [encoder setTexture:(id<MTLTexture>)manager.GetRawIndirect().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetNormal().texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:2];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorMaskBuffer() offset:0 atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:7];
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:8];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:5 withBarrier:YES];
    }
    
    // Pass 3: Accumulation
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:6 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 3: Accumulate"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoAccumulate];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(currIdx).texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetRawIndirect().texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(prevIdx).texture atIndex:2];
    [encoder setTexture:(id<MTLTexture>)manager.GetMotion().texture atIndex:3];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:4];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(prevIdx).texture atIndex:5];
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:6];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:7 withBarrier:YES];
    }
    
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    // Pass 4: Denoise (3 iterations)
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:8 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 4: Denoise"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoDenoise];
    [encoder setTexture:(id<MTLTexture>)manager.GetNormal().texture atIndex:2];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:3];
    
    int stepWidth = 1;
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoiseTemp().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(currIdx).texture atIndex:1];
    [encoder setBytes:&stepWidth length:sizeof(int) atIndex:0];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    stepWidth = 2;
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoised().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoiseTemp().texture atIndex:1];
    [encoder setBytes:&stepWidth length:sizeof(int) atIndex:0];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    stepWidth = 4;
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoiseTemp().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoised().texture atIndex:1];
    [encoder setBytes:&stepWidth length:sizeof(int) atIndex:0];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:9 withBarrier:YES];
    }
    
    // Pass 5: Volumetric
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:10 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 5: Volumetric"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoVolumetric];
    [encoder setTexture:(id<MTLTexture>)manager.GetVolumetric(currIdx).texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetVolumetric(prevIdx).texture atIndex:2];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorMaskBuffer() offset:0 atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:7];
    [encoder dispatchThreads:gridSizeHalf threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:11 withBarrier:YES];
    }
    
    // Pass 6: Exposure
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:12 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 6: Exposure"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoExposure];
    [encoder setBuffer:(id<MTLBuffer>)_exposureBuffer offset:0 atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(currIdx).texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetAlbedo().texture atIndex:2];
    [encoder dispatchThreads:groupSize threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:13 withBarrier:YES];
    }
    
    // Pass 7: Composite
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:14 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 7: Composite"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoComposite];
    [encoder setTexture:(id<MTLTexture>)manager.GetCompositeResult().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoiseTemp().texture atIndex:2];
    [encoder setTexture:(id<MTLTexture>)manager.GetAlbedo().texture atIndex:3];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:4];
    [encoder setTexture:(id<MTLTexture>)manager.GetVolumetric(currIdx).texture atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)_exposureBuffer offset:0 atIndex:0];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:15 withBarrier:YES];
    }
    
    // Pass 8: Text Overlay
    if (_fontAtlas.IsValid() && _textRenderer.GetNumGlyphs() > 0) {
        [encoder pushDebugGroup:@"Pass 8: TextOverlay"];
        [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoTextOverlay];
        [encoder setTexture:(id<MTLTexture>)manager.GetCompositeResult().texture atIndex:0];
        [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:1];
        [encoder setTexture:(id<MTLTexture>)_textRenderer.GetAtlasTexture() atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)_textRenderer.GetGlyphBuffer() offset:0 atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)_textRenderer.GetOverlayDataBuffer() offset:0 atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)_textRenderer.GetTileBuffer() offset:0 atIndex:2];
        [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
        [encoder popDebugGroup];
    }
    
    [encoder endEncoding];
    
    // Timestamp before MetalFX
    if (_supportsTimestamps) {
        id<MTLBlitCommandEncoder> preFX = [cmdBuf blitCommandEncoder];
        [preFX sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:16 withBarrier:NO];
        [preFX endEncoding];
    }
    
    // MetalFX Temporal Scaler
    id<MTLFXTemporalScaler_Unlocked> scaler = (id<MTLFXTemporalScaler_Unlocked>)_temporalScaler;
    scaler.motionVectorScale = simd_make_float2(-(float)State::dispWIDTH, -(float)State::dispHEIGHT);
    scaler.jitterOffset = simd_make_float2(-character.jitterX, -character.jitterY);
    
    ((id<MTLFXTemporalScaler>)_temporalScaler).colorTexture = (id<MTLTexture>)manager.GetCompositeResult().texture;
    ((id<MTLFXTemporalScaler>)_temporalScaler).depthTexture = (id<MTLTexture>)manager.GetDepth(currIdx).texture;
    ((id<MTLFXTemporalScaler>)_temporalScaler).motionTexture = (id<MTLTexture>)manager.GetMotion().texture;
    ((id<MTLFXTemporalScaler>)_temporalScaler).outputTexture = (id<MTLTexture>)manager.GetFinal().texture;
    ((id<MTLFXTemporalScaler>)_temporalScaler).reset = _scalerNeedsReset;
    
    if (_temporalScaler) {
        [(id<MTLFXTemporalScaler>)_temporalScaler encodeToCommandBuffer:cmdBuf];
    } else {
        NSLog(@"WARNING: No temporal scaler!");
    }
    
    // Timestamp after MetalFX
    if (_supportsTimestamps) {
        id<MTLBlitCommandEncoder> postFX = [cmdBuf blitCommandEncoder];
        [postFX sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:17 withBarrier:YES];
        [postFX endEncoding];
    }
    
    _frameIndex++;
    const_cast<Character&>(character).lastRenderedViewProjectionMatrix = character.unjitteredViewProjectionMatrix;
    _scalerNeedsReset = false;
}

void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called directly. This path is deprecated.");
}

id MetalRenderer::GetOutputTexture() {
    return _renderTargetManager.GetFinal().texture;
}

void MetalRenderer::GenerateWorld() {
    _materialMap.GenerateDynamic();
}
