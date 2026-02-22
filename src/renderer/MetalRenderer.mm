#import <MetalKit/MetalKit.h>
#include <MetalFX/MetalFX.h>
#include "renderer/MetalRenderer.hpp"
#include "State.hpp"
#include "Character.hpp"
#include "renderer/ShaderTypes.h"

#include "renderer/MetalDevice.hpp"
#include "renderer/MetalBuffer.hpp"

#include "renderer/MaterialMap.hpp"

#include "cumath.h"

#include "Texturepack.h"
#include <cassert>
#include <chrono>

// Define the MetalFX protocol to allow property access
@protocol MTLFXTemporalScaler_Unlocked <NSObject>
@property (readwrite, nonatomic) simd_float2 motionVectorScale;
@property (readwrite, nonatomic) simd_float2 jitterOffset;
@end

MetalRenderer::MetalRenderer(id device_id) : _texturepack() 
{
    _device = (id<MTLDevice>)device_id;
    NSError *error = nil;

    // 1. Load the Default Library
    id<MTLLibrary> lib = [_device newDefaultLibrary];
    if (!lib) {
        NSLog(@"FATAL: Could not load default.metallib. Ensure shaders are compiled.");
        abort();
    }

    // 2. Load Rendering Kernels
    // Ensure your .metal function names match these strings
    _psoDistApprox = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"distApproximationKernel"] error:&error];
    if (!_psoDistApprox) { NSLog(@"CRITICAL ERROR: Could not find function '_psoDistApprox' in default.metallib"); abort(); }
    _psoGBuffer    = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"GBufferAndDirectLight"] error:&error];
    if (!_psoGBuffer) { NSLog(@"CRITICAL ERROR: Could not find function '_psoGBuffer' in default.metallib"); abort(); }
    _psoIndirect   = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"IndirectBounce"] error:&error];
    if (!_psoIndirect) { NSLog(@"CRITICAL ERROR: Could not find function '_psoIndirect' in default.metallib"); abort(); }
    _psoAccumulate = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"TemporalAccumulation"] error:&error];
    if (!_psoAccumulate) { NSLog(@"CRITICAL ERROR: Could not find function '_psoAccumulate' in default.metallib"); abort(); }
    _psoDenoise    = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"BilateralDenoise"] error:&error];
    if (!_psoDenoise) { NSLog(@"CRITICAL ERROR: Could not find function '_psoDenoise' in default.metallib"); abort(); }
    _psoComposite  = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"Composite"] error:&error];
    if (!_psoComposite) { NSLog(@"CRITICAL ERROR: Could not find function '_psoComposite' in default.metallib"); abort(); }
    _psoVolumetric = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"VolumetricFog"] error:&error];
    if (!_psoVolumetric) { NSLog(@"CRITICAL ERROR: Could not find function '_psoVolumetric' in default.metallib"); abort(); }
    _psoExposure   = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ComputeExposure"] error:&error];
    if (!_psoExposure) { NSLog(@"CRITICAL ERROR: Could not find function '_psoExposure' in default.metallib"); abort(); }

    if (!_psoDistApprox || !_psoGBuffer || !_psoIndirect || !_psoAccumulate || !_psoDenoise || !_psoComposite || !_psoVolumetric || !_psoExposure) {
        NSLog(@"FATAL: Failed to load rendering kernels. Error: %@", error);
        abort();
    }

    // 3. Initialize Exposure Buffer
    float initialLum = 0.5f;
    ExposureData expData;
    expData.sceneLuminance = initialLum;
    _exposureBuffer = [_device newBufferWithBytes:&expData length:sizeof(ExposureData) options:MTLResourceStorageModeShared];

    // 4. Generate World (New XBrickMap Generation)
    NSLog(@"Starting Dynamic World Generation (XBrickMap)...");
    _materialMap.GenerateDynamic();
    NSLog(@"World Generation Complete.");

    // 5. Initialize Render Targets
    createRenderTarget(State::dispWIDTH, State::dispHEIGHT);

    // 6. Setup Timestamps
    _supportsTimestamps = [((id<MTLDevice>)_device) supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    if (_supportsTimestamps) {
        id<MTLDevice> dev = (id<MTLDevice>)_device;
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> set in [dev counterSets]) {
            if ([set.name caseInsensitiveCompare:@"timestamp"] == NSOrderedSame) {
                timestampSet = set;
                break;
            }
        }
        if (timestampSet) {
            MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
            desc.counterSet = timestampSet;
            desc.label = @"TimestampCounter";
            desc.sampleCount = 18;
            desc.storageMode = MTLStorageModePrivate;
            _counterSampleBuffer = [dev newCounterSampleBufferWithDescriptor:desc error:nil];
            _timestampBuffer = [dev newBufferWithLength:18 * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        }
    }
}

void MetalRenderer::ResetScaler() {
    _scalerNeedsReset = true;
}

MetalRenderer::~MetalRenderer() {}

void MetalRenderer::createRenderTarget(uint32_t width, uint32_t height) 
{
    auto makeTex = [&](MTLPixelFormat fmt, NSString* label) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:width height:height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [_device newTextureWithDescriptor:d];
        t.label = label;
        return t;
    };

    _texDirectLight = nil;
    _texAlbedo = nil;
    _texNormal = nil;
    _texMotion = nil;
    _texRawIndirect = nil;
    _texDenoised = nil;
    _texFinal = nil;
    _halfDistTexture = nil;
    for(int i=0; i<2; i++) {
        _texDepth[i] = nil;
        _texAccum[i] = nil;
        _texFinalHistory[i] = nil;
    }

    // G-Buffer Textures
    _texDirectLight = makeTex(MTLPixelFormatRGBA16Float, @"DirectLight");
    _texAlbedo      = makeTex(MTLPixelFormatRGBA8Unorm,  @"Albedo");
    _texNormal      = makeTex(MTLPixelFormatRGBA8Snorm, @"Normal");
    _texMotion      = makeTex(MTLPixelFormatRG16Float,   @"Motion");
    _texRawIndirect = makeTex(MTLPixelFormatRGBA16Float, @"RawIndirect");
    _texDenoised    = makeTex(MTLPixelFormatRGBA16Float, @"Denoised");
    _texFinal       = makeTex(MTLPixelFormatRGBA8Unorm,  @"FinalOutput");
    _texDenoiseTemp = makeTex(MTLPixelFormatRGBA16Float, @"DenoiseTemp");

    for(int i=0; i<2; i++) {
        _texDepth[i] = makeTex(MTLPixelFormatR32Float, [NSString stringWithFormat:@"Depth_%d", i]);
        _texAccum[i] = makeTex(MTLPixelFormatRGBA16Float, [NSString stringWithFormat:@"Accum_%d", i]);
        _texFinalHistory[i] = makeTex(MTLPixelFormatRGBA16Float, [NSString stringWithFormat:@"FinalHistory_%d", i]);
    }

    // Half-res Distance Texture
    MTLTextureDescriptor *distDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:width/2 height:height/2 mipmapped:NO];
    distDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    distDesc.storageMode = MTLStorageModePrivate;
    _halfDistTexture = [_device newTextureWithDescriptor:distDesc];

    // Half-res Volumetric Texture
    MTLTextureDescriptor *volDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:width / 2 height:height / 2 mipmapped:NO];
    volDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    volDesc.storageMode = MTLStorageModePrivate;

    _texVolumetric[0] = [_device newTextureWithDescriptor:volDesc];
    ((id<MTLTexture>)_texVolumetric[0]).label = @"Volumetric_0";
    _texVolumetric[1] = [_device newTextureWithDescriptor:volDesc];
    ((id<MTLTexture>)_texVolumetric[1]).label = @"Volumetric_1";

    // Composite Result (Input to MetalFX)
    MTLTextureDescriptor *compDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:width height:height mipmapped:NO];
    compDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    compDesc.storageMode = MTLStorageModePrivate;
    _texCompositeResult = [_device newTextureWithDescriptor:compDesc];
    [(id<MTLTexture>)_texCompositeResult setLabel:@"CompositeResult(Aliased)"];

    // Initialize MetalFX Temporal Scaler
    MTLFXTemporalScalerDescriptor* scalerDesc = [[MTLFXTemporalScalerDescriptor alloc] init];
    scalerDesc.inputWidth = width;
    scalerDesc.inputHeight = height;
    scalerDesc.outputWidth = width; // 1.0x scaling (TAA)
    scalerDesc.outputHeight = height;
    scalerDesc.colorTextureFormat = MTLPixelFormatRGBA16Float; 
    scalerDesc.depthTextureFormat = MTLPixelFormatR32Float;    
    scalerDesc.motionTextureFormat = MTLPixelFormatRG16Float;  
    scalerDesc.outputTextureFormat = MTLPixelFormatRGBA8Unorm; 
    
    _temporalScaler = [scalerDesc newTemporalScalerWithDevice:_device];
    _scalerNeedsReset = true;
}

void MetalRenderer::OnResize(uint32_t newWidth, uint32_t newHeight)
{
    State::dispWIDTH = newWidth;
    State::dispHEIGHT = newHeight;
    State::screenHEIGHT = newWidth; // Note: Original code had height/width swapped here? Keeping consistent with original.
    State::screenWIDTH = newHeight;

    if ([(id<MTLTexture>)_texFinal width] != newWidth || [(id<MTLTexture>)_texFinal height] != newHeight) {
        createRenderTarget(newWidth, newHeight);
    }
}

void MetalRenderer::Draw(id<MTLCommandBuffer> cmdBuf, const Character& character, unsigned int frameCount)
{
    // --- 1. Data Preparation ---
    int currIdx = _frameIndex % 2;
    int prevIdx = (_frameIndex + 1) % 2;

    // Camera Data Setup
    CameraData camData;
    camData.position = { (float)character.position.x, (float)character.position.y, (float)character.position.z };
    camData.forward  = { (float)character.direction.x, (float)character.direction.y, (float)character.direction.z };
    
    float tanHalfFov = tan(glm::radians(character.FOV) * 0.5f);
    float aspect = (float)State::dispWIDTH / (float)State::dispHEIGHT;
    glm::vec3 sRight = character.camera.right * tanHalfFov * aspect;
    glm::vec3 sUp    = character.camera.up * tanHalfFov;
    
    camData.right  = { sRight.x, sRight.y, sRight.z };
    camData.up     = { sUp.x,    sUp.y,    sUp.z };
    camData.jitter = { character.jitterX, character.jitterY };
    memcpy(&camData.unjitteredViewProjection, &character.unjitteredViewProjectionMatrix, 64);
    memcpy(&camData.prevUnjitteredViewProjection, &character.lastRenderedViewProjectionMatrix, 64);

    // Frame Data Setup
    FrameData frameData;
    frameData.sunDirection = simd_normalize(simd_make_float3(10.f, 5.f, -4.f));
    double time = CFAbsoluteTimeGetCurrent();
    frameData.time = (float)fmod(time, 3600.0);
    static double lastTime = time;
    frameData.deltaTime = max((float)(time - lastTime), 0.001f);
    lastTime = time;

    // --- 2. Compute Encoding Start ---
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"Render Pipeline";

    MTLSize gridSizeFull = MTLSizeMake([(id<MTLTexture>)_texFinal width], [(id<MTLTexture>)_texFinal height], 1);
    MTLSize gridSizeHalf = MTLSizeMake(gridSizeFull.width / 2, gridSizeFull.height / 2, 1);
    MTLSize groupSize    = MTLSizeMake(16, 16, 1);

    // -----------------------------------------------------------
    // PASS 0: Distance Accelerator (Approximate SDF)
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:0 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 0: Approx"];
    [encoder setComputePipelineState:_psoDistApprox];
    
    // Textures
    [encoder setTexture:_halfDistTexture atIndex:0];
    
    // Uniforms
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    
    // XBrickMap Geometry Bindings
    // Note: Indices here must match your new DistApprox shader signature
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:2]; 
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:5];

    [encoder dispatchThreads:gridSizeHalf threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:1 withBarrier:YES];
    
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------------------
    // PASS 1: G-Buffer (Primary Rays)
    // -----------------------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:2 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 1: GBuffer"];
    [encoder setComputePipelineState:_psoGBuffer];
    
    // Outputs
    [encoder setTexture:_texDirectLight atIndex:0];
    [encoder setTexture:_texAlbedo atIndex:1];
    [encoder setTexture:_texNormal atIndex:2];
    [encoder setTexture:_texMotion atIndex:3];
    [encoder setTexture:_texDepth[currIdx] atIndex:4];
    
    // Uniforms
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];

    // XBrickMap Bindings
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:5];

    // Assets & Helpers
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:8]; // Atlas
    [encoder setTexture:_halfDistTexture atIndex:9]; // Approx Dist

    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:3 withBarrier:YES];

    // -----------------------------------------------------------------------
    // PASS 2: Indirect (Bounce Rays)
    // -----------------------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:4 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 2: Indirect"];
    [encoder setComputePipelineState:_psoIndirect];
    
    // IO
    [encoder setTexture:_texRawIndirect atIndex:0];
    [encoder setTexture:_texNormal atIndex:1];
    [encoder setTexture:_texDepth[currIdx] atIndex:2];
    
    // Uniforms
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];

    // XBrickMap Bindings
    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetDataBuffer() offset:0 atIndex:5];
    
    // Assets
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:8];

    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:5 withBarrier:YES];

    // -----------------------------------------------------------
    // PASS 3: Temporal Accumulation
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:6 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 3: Accumulate"];
    [encoder setComputePipelineState:_psoAccumulate];
    [encoder setTexture:_texAccum[currIdx] atIndex:0];
    [encoder setTexture:_texRawIndirect atIndex:1];
    [encoder setTexture:_texAccum[prevIdx] atIndex:2];
    [encoder setTexture:_texMotion atIndex:3];
    [encoder setTexture:_texDepth[currIdx] atIndex:4];
    [encoder setTexture:_texDepth[prevIdx] atIndex:5];
    [encoder setTexture:_texDirectLight atIndex:6];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:7 withBarrier:YES];

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 4: Spatial Denoising (A-Trous)
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:8 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 4: Denoise"];
    [encoder setComputePipelineState:_psoDenoise];
    [encoder setTexture:_texNormal atIndex:2];
    [encoder setTexture:_texDepth[currIdx] atIndex:3];

    id<MTLTexture> inputTex = _texAccum[currIdx];
    id<MTLTexture> outputTex = _texDenoiseTemp;

    for(int i = 0; i < 3; i++) {
        int stepWidth = 1 << i; 
        [encoder setTexture:outputTex atIndex:0];
        [encoder setTexture:inputTex atIndex:1];
        [encoder setBytes:&stepWidth length:sizeof(int) atIndex:0];
        [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
        [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
        
        // Swap
        id<MTLTexture> temp = inputTex; inputTex = outputTex;
        // Last iteration writes to final Denoised texture
        outputTex = (i == 1) ? _texDenoised : temp; 
    }
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:9 withBarrier:YES];

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 5: Volumetric Fog
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:10 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 5: Volumetric"];
    [encoder setComputePipelineState:_psoVolumetric];
    
    // IO
    [encoder setTexture:_texVolumetric[currIdx] atIndex:0]; 
    [encoder setTexture:_texDepth[currIdx] atIndex:1];
    [encoder setTexture:_texVolumetric[prevIdx] atIndex:2]; // History
    
    // Uniforms
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];


    [encoder setTexture:(id<MTLTexture>)_materialMap.GetIndirectionTexture() atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetSectorBuffer() offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)_materialMap.GetOccupancyBuffer() offset:0 atIndex:4];


    [encoder dispatchThreads:gridSizeHalf threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:11 withBarrier:YES];

    // -----------------------------------------------------------
    // PASS 6: Auto-Exposure
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:12 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 6: Exposure"];
    [encoder setComputePipelineState:_psoExposure];
    [encoder setBuffer:_exposureBuffer offset:0 atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_texDirectLight atIndex:0];
    [encoder setTexture:_texAccum[currIdx] atIndex:1]; 
    [encoder setTexture:_texAlbedo atIndex:2];
    [encoder dispatchThreads:groupSize threadsPerThreadgroup:groupSize]; // Single dispatch
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:13 withBarrier:YES];

    // -----------------------------------------------------------
    // PASS 7: Composite
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:14 withBarrier:NO];
    [encoder pushDebugGroup:@"Pass 7: Composite"];
    [encoder setComputePipelineState:_psoComposite];
    [encoder setTexture:_texCompositeResult atIndex:0]; 
    [encoder setTexture:_texDirectLight atIndex:1];
    [encoder setTexture:_texDenoised atIndex:2];
    [encoder setTexture:_texAlbedo atIndex:3];
    [encoder setTexture:_texDepth[currIdx] atIndex:4];
    [encoder setTexture:_texVolumetric[currIdx] atIndex:5];
    [encoder setBuffer:_exposureBuffer offset:0 atIndex:0];
    [encoder dispatchThreads:gridSizeFull threadsPerThreadgroup:groupSize];
    [encoder popDebugGroup];
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:15 withBarrier:YES];

    [encoder endEncoding]; 

    // -----------------------------------------------------------
    // PASS 8: MetalFX Upscaling
    // -----------------------------------------------------------
    if (_supportsTimestamps) {
        id<MTLBlitCommandEncoder> preFX = [cmdBuf blitCommandEncoder];
        [preFX sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:16 withBarrier:NO];
        [preFX endEncoding];
    }

    // Config Scaler
    id<MTLFXTemporalScaler_Unlocked> scaler = (id<MTLFXTemporalScaler_Unlocked>)_temporalScaler;
    scaler.motionVectorScale = simd_make_float2(-(float)State::dispWIDTH, -(float)State::dispHEIGHT);
    scaler.jitterOffset = simd_make_float2(-character.jitterX, -character.jitterY);
    
    _temporalScaler.colorTexture = (id<MTLTexture>)_texCompositeResult;
    _temporalScaler.depthTexture = (id<MTLTexture>)_texDepth[currIdx];
    _temporalScaler.motionTexture = (id<MTLTexture>)_texMotion;
    _temporalScaler.outputTexture = (id<MTLTexture>)_texFinal;
    _temporalScaler.reset = _scalerNeedsReset;
    _scalerNeedsReset = false;

    [_temporalScaler encodeToCommandBuffer:cmdBuf];

    if (_supportsTimestamps) {
        id<MTLBlitCommandEncoder> postFX = [cmdBuf blitCommandEncoder];
        [postFX sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:17 withBarrier:YES];
        [postFX endEncoding];
    }
    
    _frameIndex++;

    const_cast<Character&>(character).lastRenderedViewProjectionMatrix = character.unjitteredViewProjectionMatrix;
}

void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called directly. This path is deprecated.");
}

id MetalRenderer::GetOutputTexture()
{
    return _texFinal;
}

void MetalRenderer::GenerateWorld() {
    _materialMap.GenerateDynamic();
}

void MetalRenderer::generateNoiseTexture() {}