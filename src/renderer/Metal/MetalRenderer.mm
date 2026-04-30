#include "renderer/Metal/MetalRenderer.hpp"
#include "Character.hpp"
#include "State.hpp"
#include "Texturepack.h"
#include "cumath.h"
#include "renderer/MaterialMap.hpp"
#include "renderer/ShaderTypes.h"
#include "renderer/ShaderGlobalParams.hpp"
#include "renderer/shader_settings.h"
#include "console/GameConsole.hpp"
#include "console/ConsoleBuffer.hpp"
#include "ShaderBindings.generated.hpp"
#include <MetalFX/MetalFX.h>
#import <MetalKit/MetalKit.h>
#include <cassert>
#include <chrono>

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
    , _outputTexture(nullptr)
    , _computeToMetalFXFence(nullptr)
    , _counterSampleBuffer(nullptr)
    , _timestampBuffer(nullptr)
    , _supportsTimestamps(false)
    , _psoTextOverlay(nullptr)
{
    // Create command queue
    _commandQueue = [device newCommandQueue];
    
    // Create synchronization fence for compute-to-MetalFX handoff
    _computeToMetalFXFence = [(id<MTLDevice>)device newFence];
    ((id<MTLFence>)_computeToMetalFXFence).label = @"ComputeToMetalFX";
    
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

    // Create shared linear sampler
    {
        MTLSamplerDescriptor *samplerDesc = [[MTLSamplerDescriptor alloc] init];
        samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
        samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
        samplerDesc.mipFilter = MTLSamplerMipFilterNotMipmapped;
        samplerDesc.sAddressMode = MTLSamplerAddressModeRepeat;
        samplerDesc.tAddressMode = MTLSamplerAddressModeRepeat;
        samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
        _linearSampler = [(id<MTLDevice>)device newSamplerStateWithDescriptor:samplerDesc];
    }

// Initialize render targets
  Initialize(State::dispWIDTH, State::dispHEIGHT);
  
  // Create output texture at screen resolution for MetalFX / presentation
  CreateOutputTexture(State::screenWIDTH, State::screenHEIGHT);
  
  // Create MetalFX temporal scaler (input=render res, output=screen res)
  CreateTemporalScaler(State::dispWIDTH, State::dispHEIGHT, State::screenWIDTH, State::screenHEIGHT);
    
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
    _psoBilateralUpsample = [device newComputePipelineStateWithFunction:
                               [_library newFunctionWithName:@"BilateralUpsample"]
                                                                    error:&error];
    _psoFallbackBlit = [device newComputePipelineStateWithFunction:
                               [_library newFunctionWithName:@"FallbackBlit"]
                                                                    error:&error];
    
    if (!_psoDistApprox || !_psoGBuffer || !_psoIndirect || !_psoAccumulate ||
        !_psoDenoise || !_psoComposite || !_psoVolumetric || !_psoExposure || !_psoTextOverlay || !_psoBilateralUpsample || !_psoFallbackBlit) {
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
    _psoBilateralUpsample = nullptr;
    _psoFallbackBlit = nullptr;
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
            desc.sampleCount = 20;
            desc.storageMode = MTLStorageModePrivate;
            _counterSampleBuffer = [device newCounterSampleBufferWithDescriptor:desc error:nil];
            _timestampBuffer = [device newBufferWithLength:20 * sizeof(uint64_t)
                                                    options:MTLResourceStorageModeShared];
        }
    }
}

void MetalRenderer::CreateTemporalScaler(uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight) {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    
#if USE_METALFX
    MTLFXTemporalScalerDescriptor *scalerDesc = [[MTLFXTemporalScalerDescriptor alloc] init];
    scalerDesc.inputWidth = renderWidth;
    scalerDesc.inputHeight = renderHeight;
    scalerDesc.outputWidth = outputWidth;
    scalerDesc.outputHeight = outputHeight;
    scalerDesc.colorTextureFormat = MTLPixelFormatRGBA16Float;
    scalerDesc.depthTextureFormat = MTLPixelFormatR32Float;
    scalerDesc.motionTextureFormat = MTLPixelFormatRG16Float;
    scalerDesc.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
    
    _temporalScaler = [scalerDesc newTemporalScalerWithDevice:device];
    _scalerNeedsReset = true;
    
    NSLog(@"MetalFX Temporal Scaler created: %dx%d -> %dx%d", renderWidth, renderHeight, outputWidth, outputHeight);
#else
    _temporalScaler = nullptr;
    _scalerNeedsReset = false;
    NSLog(@"MetalFX Temporal Scaler is DISABLED via USE_METALFX fallback");
#endif
}

void MetalRenderer::CreateOutputTexture(uint32_t width, uint32_t height) {
    id<MTLDevice> device = (id<MTLDevice>)_device;
    
    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2D;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.width = width;
    desc.height = height;
    desc.mipmapLevelCount = 1;
    desc.sampleCount = 1;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModePrivate;
    
    _outputTexture = [device newTextureWithDescriptor:desc];
    NSLog(@"Output texture created at %dx%d", width, height);
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
    clearTexture((id<MTLTexture>)manager.GetRawIndirectHalf().texture, MTLPixelFormatRGBA16Float);
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
    auto drawStartTime = std::chrono::high_resolution_clock::now();

    int currIdx = _frameIndex % 2;
    int prevIdx = (_frameIndex + 1) % 2;

    // Prepare data using FrameDataManager
    simd_int3 worldOrigin = _materialMap.GetWorldOrigin();
    CameraData camData = _frameDataManager.PrepareCameraData(character);
    FrameData frameData = _frameDataManager.PrepareFrameData(_frameIndex, worldOrigin);
    CharacterGPUData charData = _frameDataManager.PrepareCharacterData(
        character, State::state.otherCharacters);

    // Update material map streaming
    auto streamingStart = std::chrono::high_resolution_clock::now();
    simd_float3 camPos = camData.position;
    bool sectorsChanged = _materialMap.UpdateStreaming(camPos);
    if (sectorsChanged) {
        _scalerNeedsReset = true;
    }
    auto streamingEnd = std::chrono::high_resolution_clock::now();
    cpuStreamingMs = std::chrono::duration<double, std::milli>(streamingEnd - streamingStart).count();

    // Prepare text overlay (console rendering)
    auto textPrepStart = std::chrono::high_resolution_clock::now();
    if (_fontAtlas.IsValid()) {
        _textRenderer.BeginFrame(State::dispWIDTH, State::dispHEIGHT);
        RenderConsole();
        _textRenderer.EndFrame();
        _textRenderer.UpdateBuffers((id<MTLDevice>)_device);
    }
    auto textPrepEnd = std::chrono::high_resolution_clock::now();
    cpuTextPrepMs = std::chrono::duration<double, std::milli>(textPrepEnd - textPrepStart).count();
    
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
    {
        GlobalParams gp = { camData, frameData };
        [encoder setBytes:&gp length:sizeof(GlobalParams) atIndex:ShaderBindings::distApproximationKernel::buffer::globalParams];
    }
    [encoder setTexture:(id<MTLTexture>)manager.GetHalfDist().texture atIndex:ShaderBindings::distApproximationKernel::texture::distTex];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:ShaderBindings::distApproximationKernel::texture::indirection];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:ShaderBindings::distApproximationKernel::buffer::sectorBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:ShaderBindings::distApproximationKernel::buffer::charData];
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
    {
        GlobalParams gp = { camData, frameData };
        [encoder setBytes:&gp length:sizeof(GlobalParams) atIndex:ShaderBindings::GBufferAndDirectLight::buffer::globalParams];
    }
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:ShaderBindings::GBufferAndDirectLight::texture::texDirectLight];
    [encoder setTexture:(id<MTLTexture>)manager.GetAlbedo().texture atIndex:ShaderBindings::GBufferAndDirectLight::texture::texAlbedo];
    [encoder setTexture:(id<MTLTexture>)manager.GetNormal().texture atIndex:ShaderBindings::GBufferAndDirectLight::texture::texNormal];
    [encoder setTexture:(id<MTLTexture>)manager.GetMotion().texture atIndex:ShaderBindings::GBufferAndDirectLight::texture::texMotion];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:ShaderBindings::GBufferAndDirectLight::texture::texDepth];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:ShaderBindings::GBufferAndDirectLight::texture::indirection];
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:ShaderBindings::GBufferAndDirectLight::texture::textureAtlas];
    [encoder setTexture:(id<MTLTexture>)manager.GetHalfDist().texture atIndex:ShaderBindings::GBufferAndDirectLight::texture::halfDistTex];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:ShaderBindings::GBufferAndDirectLight::buffer::sectorBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:ShaderBindings::GBufferAndDirectLight::buffer::occupancyBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:ShaderBindings::GBufferAndDirectLight::buffer::dataBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:ShaderBindings::GBufferAndDirectLight::buffer::charData];
    [encoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::GBufferAndDirectLight::sampler::atlasSampler];
    [encoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::GBufferAndDirectLight::sampler::distSampler];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:3 withBarrier:YES];
    }
    
    // Ensure GBuffer outputs (Normal, Depth) are visible to Pass 2
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    // Pass 2: Indirect (half-resolution)
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:4 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 2: Indirect"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoIndirect];
    {
        GlobalParams gp = { camData, frameData };
        [encoder setBytes:&gp length:sizeof(GlobalParams) atIndex:ShaderBindings::IndirectBounce::buffer::globalParams];
    }
    [encoder setTexture:(id<MTLTexture>)manager.GetRawIndirectHalf().texture atIndex:ShaderBindings::IndirectBounce::texture::texRawIndirect];
    [encoder setTexture:(id<MTLTexture>)manager.GetNormal().texture atIndex:ShaderBindings::IndirectBounce::texture::texNormal];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:ShaderBindings::IndirectBounce::texture::texDepth];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:ShaderBindings::IndirectBounce::texture::indirection];
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:ShaderBindings::IndirectBounce::texture::textureAtlas];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:ShaderBindings::IndirectBounce::buffer::sectorBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:ShaderBindings::IndirectBounce::buffer::occupancyBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:ShaderBindings::IndirectBounce::buffer::dataBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:ShaderBindings::IndirectBounce::buffer::charData];
    [encoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::IndirectBounce::sampler::atlasSampler];
    [encoder dispatchThreads:gridSizeHalf threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:5 withBarrier:YES];
    }
    
    // Pass 2.5: Bilateral Upsample (half-res indirect → full-res)
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:5 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 2.5: Upsample"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoBilateralUpsample];
    [encoder setTexture:(id<MTLTexture>)manager.GetRawIndirect().texture atIndex:0];
    [encoder setTexture:(id<MTLTexture>)manager.GetRawIndirectHalf().texture atIndex:1];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:2];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    
    // Pass 3: Accumulation
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:6 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 3: Accumulate"];
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoAccumulate];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(currIdx).texture atIndex:ShaderBindings::TemporalAccumulation::texture::texAccum];
    [encoder setTexture:(id<MTLTexture>)manager.GetRawIndirect().texture atIndex:ShaderBindings::TemporalAccumulation::texture::texRawIndirect];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(prevIdx).texture atIndex:ShaderBindings::TemporalAccumulation::texture::texHistory];
    [encoder setTexture:(id<MTLTexture>)manager.GetMotion().texture atIndex:ShaderBindings::TemporalAccumulation::texture::texMotion];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:ShaderBindings::TemporalAccumulation::texture::texDepth];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(prevIdx).texture atIndex:ShaderBindings::TemporalAccumulation::texture::texPrevDepth];
    [encoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::TemporalAccumulation::sampler::historySampler];
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
    {
        GlobalParams gp = { camData, frameData };
        [encoder setBytes:&gp length:sizeof(GlobalParams) atIndex:ShaderBindings::VolumetricFog::buffer::globalParams];
    }
    [encoder setTexture:(id<MTLTexture>)manager.GetVolumetric(currIdx).texture atIndex:ShaderBindings::VolumetricFog::texture::texVolumetric];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:ShaderBindings::VolumetricFog::texture::texDepth];
    [encoder setTexture:(id<MTLTexture>)manager.GetVolumetric(prevIdx).texture atIndex:ShaderBindings::VolumetricFog::texture::texHistory];
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:ShaderBindings::VolumetricFog::texture::indirection];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:ShaderBindings::VolumetricFog::buffer::sectorBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:ShaderBindings::VolumetricFog::buffer::occupancyBuffer];
    [encoder setBuffer:(id<MTLBuffer>)_characterBuffer offset:0 atIndex:ShaderBindings::VolumetricFog::buffer::charData];
    [encoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::VolumetricFog::sampler::historySampler];
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
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:ShaderBindings::ComputeExposure::buffer::globalParams];
    [encoder setBuffer:(id<MTLBuffer>)_exposureBuffer offset:0 atIndex:ShaderBindings::ComputeExposure::buffer::exposure];
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:ShaderBindings::ComputeExposure::texture::texDirect];
    [encoder setTexture:(id<MTLTexture>)manager.GetAccum(currIdx).texture atIndex:ShaderBindings::ComputeExposure::texture::texAccum];
    [encoder setTexture:(id<MTLTexture>)manager.GetAlbedo().texture atIndex:ShaderBindings::ComputeExposure::texture::texAlbedo];
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
    [encoder setTexture:(id<MTLTexture>)manager.GetCompositeResult().texture atIndex:ShaderBindings::Composite::texture::texFinal];
    [encoder setTexture:(id<MTLTexture>)manager.GetDirectLight().texture atIndex:ShaderBindings::Composite::texture::texDirect];
    [encoder setTexture:(id<MTLTexture>)manager.GetDenoiseTemp().texture atIndex:ShaderBindings::Composite::texture::texAccum];
    [encoder setTexture:(id<MTLTexture>)manager.GetAlbedo().texture atIndex:ShaderBindings::Composite::texture::texAlbedo];
    [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:ShaderBindings::Composite::texture::texDepth];
    [encoder setTexture:(id<MTLTexture>)manager.GetVolumetric(currIdx).texture atIndex:ShaderBindings::Composite::texture::texVolumetric];
    [encoder setBuffer:(id<MTLBuffer>)_exposureBuffer offset:0 atIndex:ShaderBindings::Composite::buffer::exposure];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:15 withBarrier:YES];
    }
    
    // Pass 8: Text Overlay
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:16 withBarrier:NO];
    }
    if (_fontAtlas.IsValid() && _textRenderer.GetNumGlyphs() > 0) {
        [encoder pushDebugGroup:@"Pass 8: TextOverlay"];
        [encoder setComputePipelineState:(id<MTLComputePipelineState>)_psoTextOverlay];
        [encoder setTexture:(id<MTLTexture>)manager.GetCompositeResult().texture atIndex:ShaderBindings::TextOverlay::texture::texComposite];
        [encoder setTexture:(id<MTLTexture>)manager.GetDepth(currIdx).texture atIndex:ShaderBindings::TextOverlay::texture::texDepth];
        [encoder setTexture:(id<MTLTexture>)_textRenderer.GetAtlasTexture() atIndex:ShaderBindings::TextOverlay::texture::texAtlas];
        [encoder setBuffer:(id<MTLBuffer>)_textRenderer.GetGlyphBuffer() offset:0 atIndex:ShaderBindings::TextOverlay::buffer::glyphs];
        [encoder setBuffer:(id<MTLBuffer>)_textRenderer.GetOverlayDataBuffer() offset:0 atIndex:ShaderBindings::TextOverlay::buffer::overlayData];
        [encoder setBuffer:(id<MTLBuffer>)_textRenderer.GetTileBuffer() offset:0 atIndex:ShaderBindings::TextOverlay::buffer::tileData];
        [encoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::TextOverlay::sampler::atlasSampler];
        [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
        [encoder popDebugGroup];
    }
    if (_supportsTimestamps) {
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:17 withBarrier:YES];
    }

    // Signal fence to mark completion of all compute work
    [encoder updateFence:(id<MTLFence>)_computeToMetalFXFence];
    [encoder endEncoding];

    // Timestamp before MetalFX
    if (_supportsTimestamps) {
        id<MTLBlitCommandEncoder> preFX = [cmdBuf blitCommandEncoder];
        [preFX sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:18 withBarrier:NO];
        [preFX endEncoding];
    }

    // Wait for compute work to complete before MetalFX reads textures
    id<MTLBlitCommandEncoder> fenceWaitEncoder = [cmdBuf blitCommandEncoder];
    fenceWaitEncoder.label = @"Fence Wait";
    [fenceWaitEncoder waitForFence:(id<MTLFence>)_computeToMetalFXFence];
    [fenceWaitEncoder endEncoding];

    if (_temporalScaler) {
        // MetalFX Temporal Scaler - feed current render as input, output to history buffer
        id<MTLFXTemporalScaler_Unlocked> scaler = (id<MTLFXTemporalScaler_Unlocked>)_temporalScaler;
        scaler.motionVectorScale = simd_make_float2(-(float)State::dispWIDTH, -(float)State::dispHEIGHT);
        scaler.jitterOffset = simd_make_float2(-character.jitterX, -character.jitterY);

        // MetalFX inputs/outputs
        ((id<MTLFXTemporalScaler>)_temporalScaler).colorTexture = (id<MTLTexture>)manager.GetCompositeResult().texture;
        ((id<MTLFXTemporalScaler>)_temporalScaler).depthTexture = (id<MTLTexture>)manager.GetDepth(currIdx).texture;
        ((id<MTLFXTemporalScaler>)_temporalScaler).motionTexture = (id<MTLTexture>)manager.GetMotion().texture;
        ((id<MTLFXTemporalScaler>)_temporalScaler).outputTexture = (id<MTLTexture>)_outputTexture;
        ((id<MTLFXTemporalScaler>)_temporalScaler).reset = _scalerNeedsReset;

        [(id<MTLFXTemporalScaler>)_temporalScaler encodeToCommandBuffer:cmdBuf];
    } else {
        // Fallback: upscale from render resolution to output resolution
        id<MTLComputeCommandEncoder> fallbackEncoder = [cmdBuf computeCommandEncoder];
        [fallbackEncoder pushDebugGroup:@"Fallback Blit"];
        [fallbackEncoder setComputePipelineState:(id<MTLComputePipelineState>)_psoFallbackBlit];
        [fallbackEncoder setTexture:(id<MTLTexture>)manager.GetCompositeResult().texture atIndex:ShaderBindings::FallbackBlit::texture::texSrc];
        [fallbackEncoder setTexture:(id<MTLTexture>)_outputTexture atIndex:ShaderBindings::FallbackBlit::texture::texDst];
        [fallbackEncoder setSamplerState:(id<MTLSamplerState>)_linearSampler atIndex:ShaderBindings::FallbackBlit::sampler::blitSampler];
        MTLSize outputGrid = MTLSizeMake(State::screenWIDTH, State::screenHEIGHT, 1);
        [fallbackEncoder dispatchThreads:outputGrid threadsPerThreadgroup:groupSize];
        [fallbackEncoder popDebugGroup];
        [fallbackEncoder endEncoding];
    }

    // Timestamp after MetalFX
    if (_supportsTimestamps) {
        id<MTLBlitCommandEncoder> postFX = [cmdBuf blitCommandEncoder];
        [postFX sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:19 withBarrier:YES];
        [postFX endEncoding];
    }

    _frameIndex++;
    _scalerNeedsReset = false;

    // Record total Draw() CPU time
    auto drawEndTime = std::chrono::high_resolution_clock::now();
    cpuDrawTotalMs = std::chrono::duration<double, std::milli>(drawEndTime - drawStartTime).count();
}

void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called directly. This path is deprecated.");
}

id MetalRenderer::GetOutputTexture() {
    return _outputTexture;
}

void MetalRenderer::OnResize(uint32_t renderW, uint32_t renderH, uint32_t screenW, uint32_t screenH) {
    RendererBase<RendererImpl::MetalRendererTraits>::OnResize(renderW, renderH, screenW, screenH);
    CreateOutputTexture(screenW, screenH);
    CreateTemporalScaler(renderW, renderH, screenW, screenH);
}

void MetalRenderer::GenerateWorld() {
    _materialMap.GenerateDynamic();
}

static simd_float4 GetConsoleMsgColor(ConsoleMsgType type, float alpha) {
    switch (type) {
        case ConsoleMsgType::System:  return simd_make_float4(1.0f, 1.0f, 1.0f, alpha);
        case ConsoleMsgType::Command: return simd_make_float4(0.7f, 0.7f, 0.7f, alpha);
        case ConsoleMsgType::Chat:    return simd_make_float4(1.0f, 1.0f, 0.5f, alpha);
        case ConsoleMsgType::Error:   return simd_make_float4(1.0f, 0.3f, 0.3f, alpha);
        default:                       return simd_make_float4(1.0f, 1.0f, 1.0f, alpha);
    }
}

void MetalRenderer::RenderConsole() {
    GameConsole& console = State::state.console;
    float screenWidth = static_cast<float>(State::dispWIDTH);
    float screenHeight = static_cast<float>(State::dispHEIGHT);
    bool isOpen = console.IsOpen();

    float lineHeight = CONSOLE_LINE_HEIGHT * CONSOLE_FONT_SCALE;
    float inputLineHeight = lineHeight + 6.0f;

    // Background rect — only when console is open
    if (isOpen) {
        float bgTop = screenHeight
                    - (CONSOLE_VISIBLE_LINES * lineHeight)
                    - inputLineHeight
                    - CONSOLE_MARGIN_BOTTOM
                    - 10.0f;
        float bgHeight = screenHeight - bgTop;

        _textRenderer.AddRect(0.0f, bgTop, screenWidth, bgHeight,
                              simd_make_float4(0.0f, 0.0f, 0.0f, CONSOLE_BG_ALPHA));
    }

    // Visible messages — always rendered (alpha differs open vs closed)
    std::vector<const ConsoleMessage*> messages;
    console.GetVisibleMessages(messages);

    float alpha = isOpen ? CONSOLE_TEXT_ALPHA : CONSOLE_TEXT_ALPHA_FADED;
    float inputOffset = isOpen ? inputLineHeight : 0.0f;
    float baseY = screenHeight - CONSOLE_MARGIN_BOTTOM - inputOffset;

    for (int i = static_cast<int>(messages.size()) - 1; i >= 0; --i) {
        float y = baseY - (static_cast<int>(messages.size()) - 1 - i) * lineHeight;
        if (y < 0.0f) break;

        simd_float4 color = GetConsoleMsgColor(messages[i]->type, alpha);
        _textRenderer.AddText(messages[i]->text,
                              CONSOLE_MARGIN_X, y,
                              CONSOLE_FONT_SCALE, color, 0.05f);
    }

    // Input line — only when console is open
    if (isOpen) {
        std::string inputText = console.GetInputDisplayText();
        float inputY = screenHeight - CONSOLE_MARGIN_BOTTOM;
        _textRenderer.AddText(inputText,
                              CONSOLE_MARGIN_X, inputY,
                              CONSOLE_FONT_SCALE,
                              simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f),
                              0.05f);
    }
}
