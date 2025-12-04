#import <MetalKit/MetalKit.h>
#include "renderer/MetalRenderer.hpp"
#include "State.hpp"
#include "Character.hpp"
#include "renderer/ShaderTypes.h"

#include "renderer/MetalDevice.hpp"
#include "renderer/MetalBuffer.hpp"
#include "cumath.h"

#include "Texturepack.h"
#include <cassert>
#include <chrono> // Ensure chrono is included

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
    _psoDistApprox = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"distApproximationKernel"] error:&error];
    _psoGBuffer    = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"GBufferAndDirectLight"] error:&error];
    _psoIndirect   = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"IndirectBounce"] error:&error];
    _psoAccumulate = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"TemporalAccumulation"] error:&error];
    _psoDenoise    = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"BilateralDenoise"] error:&error];
    _psoComposite  = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"Composite"] error:&error];

    if (!_psoDistApprox || !_psoGBuffer || !_psoIndirect || !_psoAccumulate || !_psoDenoise || !_psoComposite) {
        NSLog(@"FATAL: Failed to load rendering kernels. Error: %@", error);
        abort();
    }

    // 3. World Generation Setup
    id<MTLFunction> worldGenFunc = [lib newFunctionWithName:@"GeneratePackedWorld"];
    if (!worldGenFunc) {
        NSLog(@"FATAL: Could not find function 'GeneratePackedWorld'. Check CArray_impl.metal.");
        abort();
    }

    _worldGenerationPSO = [_device newComputePipelineStateWithFunction:worldGenFunc error:&error];
    if (!_worldGenerationPSO) {
        NSLog(@"FATAL: Failed to create World Gen PSO: %@", error);
        abort();
    }

    // Calculate dimensions
    // SHIX is 11 -> SIZEX = 2048. Packed = 512.
    // SHIZ is 11 -> SIZEZ = 2048. Packed = 1024 (since Z is packed 2 bits per block? No, bits are 32 per block).
    // Let's rely on constants defined in cumath.h
    const int packedWidth  = SIZEX / 4;
    const int packedHeight = SIZEY / 4;
    const int packedDepth  = SIZEZ / 2;

    NSLog(@"Allocating Voxel World Texture: %dx%dx%d (R32Uint)", packedWidth, packedHeight, packedDepth);

    MTLTextureDescriptor *voxelDesc = [[MTLTextureDescriptor alloc] init];
    voxelDesc.textureType = MTLTextureType3D;
    voxelDesc.pixelFormat = MTLPixelFormatR32Uint; 
    voxelDesc.width = packedWidth;
    voxelDesc.height = packedHeight;
    voxelDesc.depth = packedDepth;
    voxelDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    voxelDesc.storageMode = MTLStorageModePrivate;

    _voxelTexture = [_device newTextureWithDescriptor:voxelDesc];
    if (!_voxelTexture) {
        NSLog(@"FATAL: Failed to allocate 3D Voxel Texture (~1GB). Out of Memory?");
        abort();
    }

    // 4. Execute World Generation
    id<MTLCommandQueue> queue = [_device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:_worldGenerationPSO]; 
    [encoder setTexture:_voxelTexture atIndex:0];

    // Dispatch
    [encoder dispatchThreads:MTLSizeMake(packedWidth, packedHeight, packedDepth) 
       threadsPerThreadgroup:MTLSizeMake(8, 8, 8)];
    
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    NSLog(@"World Generation Complete.");

    // 5. Generate SDF and GI structures
    _csdf.AllocateSDF();
    _csdf.GenerateSDF((__bridge void*)_voxelTexture);
    
    // 6. Initialize Render Targets
    createRenderTarget(State::dispWIDTH, State::dispHEIGHT);

    // 7. Setup Timestamps
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
            desc.sampleCount = 12;
            desc.storageMode = MTLStorageModePrivate;
            _counterSampleBuffer = [dev newCounterSampleBufferWithDescriptor:desc error:nil];
            _timestampBuffer = [dev newBufferWithLength:12 * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        }
    }
}

MetalRenderer::~MetalRenderer() {}

void MetalRenderer::createRenderTarget(uint32_t width, uint32_t height) 
{
    // Helper lambda for standard private textures
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
    }

    // G-Buffer
    _texDirectLight = makeTex(MTLPixelFormatRGBA16Float, @"DirectLight");
    _texAlbedo      = makeTex(MTLPixelFormatRGBA8Unorm,  @"Albedo");
    _texNormal      = makeTex(MTLPixelFormatRGBA8Snorm, @"Normal");
    _texMotion      = makeTex(MTLPixelFormatRG16Float,   @"Motion");
    _texRawIndirect = makeTex(MTLPixelFormatRGBA16Float, @"RawIndirect");
    _texDenoised    = makeTex(MTLPixelFormatRGBA16Float, @"Denoised");
    _texFinal       = makeTex(MTLPixelFormatRGBA8Unorm,  @"FinalOutput");

    for(int i=0; i<2; i++) {
        _texDepth[i] = makeTex(MTLPixelFormatR32Float, [NSString stringWithFormat:@"Depth_%d", i]);
        _texAccum[i] = makeTex(MTLPixelFormatRGBA16Float, [NSString stringWithFormat:@"Accum_%d", i]);
    }

    MTLTextureDescriptor *distDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:width/2 height:height/2 mipmapped:NO];
    distDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    distDesc.storageMode = MTLStorageModePrivate;
    _halfDistTexture = [_device newTextureWithDescriptor:distDesc];



//    id<MTLCommandQueue> queue = [_device newCommandQueue];
//    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
//    MTLRenderPassDescriptor* clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
//
//    for(int i = 0; i < 2; i++) {
//        clearPass.colorAttachments[0].texture = _texAccum[i];
//        clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
//        clearPass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
//        clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
//        
//        id<MTLRenderCommandEncoder> clearEncoder = [cmdBuf renderCommandEncoderWithDescriptor:clearPass];
//        [clearEncoder endEncoding];
//    }
//    [cmdBuf commit];
//    [cmdBuf waitUntilCompleted];
}

void MetalRenderer::OnResize(uint32_t newWidth, uint32_t newHeight)
{
    State::dispWIDTH = newWidth;
    State::dispHEIGHT = newHeight;

    State::screenHEIGHT = newWidth;
    State::screenWIDTH = newHeight;
    if ([(id<MTLTexture>)_texFinal width] != newWidth || [(id<MTLTexture>)_texFinal height] != newHeight) {
        createRenderTarget(newWidth, newHeight);
    }
}

void MetalRenderer::Draw(id<MTLComputeCommandEncoder> encoder, const Character& character, unsigned int frameCount)
{
    int currIdx = _frameIndex % 2;
    int prevIdx = (_frameIndex + 1) % 2;

    // Common Data
    CameraData camData;
    camData.position = { (float)character.position.x, (float)character.position.y, (float)character.position.z };
    camData.forward  = { (float)character.direction.x, (float)character.direction.y, (float)character.direction.z };
    camData.right    = { character.camera.right.x, character.camera.right.y, character.camera.right.z };
    camData.up       = { character.camera.up.x, character.camera.up.y, character.camera.up.z };
    camData.jitter = { character.jitterX, character.jitterY };

    memcpy(&camData.unjitteredViewProjection, &character.unjitteredViewProjectionMatrix, sizeof(float) * 16);
    memcpy(&camData.prevUnjitteredViewProjection, &character.lastRenderedViewProjectionMatrix, sizeof(float) * 16);

    FrameData frameData;
    frameData.sunDirection = simd_normalize(simd_make_float3(10.f, 5.f, -4.f));

    double time = CFAbsoluteTimeGetCurrent();
    frameData.time = (float)fmod(time, 3600.0);

    // -----------------------------------------------------------
    // PASS 0: Distance Accelerator
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:0 withBarrier:NO];

    [encoder pushDebugGroup:@"Pass 0: Approx"];
    [encoder setComputePipelineState:_psoDistApprox];
    [encoder setTexture:_halfDistTexture atIndex:0];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    
    MTLSize gridHalf = MTLSizeMake([(id<MTLTexture>)_halfDistTexture width], [(id<MTLTexture>)_halfDistTexture height], 1);
    [encoder dispatchThreads:gridHalf threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder popDebugGroup];

    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:1 withBarrier:YES];

    // Texture Barrier to ensure half-dist is written
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 1: G-Buffer & Direct Light
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:2 withBarrier:NO];

    [encoder pushDebugGroup:@"Pass 1: GBuffer"];
    [encoder setComputePipelineState:_psoGBuffer];
    
    // Outputs
    [encoder setTexture:_texDirectLight atIndex:0];
    [encoder setTexture:_texAlbedo atIndex:1];
    [encoder setTexture:_texNormal atIndex:2];
    [encoder setTexture:_texMotion atIndex:3];
    [encoder setTexture:_texDepth[currIdx] atIndex:4]; // Write to CURRENT depth

    // Inputs
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:5];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:6];
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:7];
    [encoder setTexture:_halfDistTexture atIndex:8];

    MTLSize gridFull = MTLSizeMake([(id<MTLTexture>)_texFinal width], [(id<MTLTexture>)_texFinal height], 1);
    MTLSize threadGroup = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroup];
    [encoder popDebugGroup];

    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:3 withBarrier:YES];

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 2: Indirect Bounce (Path Tracing)
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:4 withBarrier:NO];

    [encoder pushDebugGroup:@"Pass 2: Indirect"];
    [encoder setComputePipelineState:_psoIndirect];
    
    // Output
    [encoder setTexture:_texRawIndirect atIndex:0];
    
    // Inputs (G-Buffer)
    [encoder setTexture:_texNormal atIndex:1];
    [encoder setTexture:_texDepth[currIdx] atIndex:2]; // Read CURRENT depth
    
    // Bind Global Data
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:3];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:4];
    [encoder setTexture:(__bridge id<MTLTexture>)_texturepack.getTextureObject() atIndex:5];
    
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroup];
    [encoder popDebugGroup];

    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:5 withBarrier:YES];

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 3: Temporal Accumulation
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:6 withBarrier:NO];

    [encoder pushDebugGroup:@"Pass 3: Accumulate"];
    [encoder setComputePipelineState:_psoAccumulate];

    // Output: Write to CURRENT Accum
    [encoder setTexture:_texAccum[currIdx] atIndex:0];

    // Inputs
    [encoder setTexture:_texRawIndirect atIndex:1];
    [encoder setTexture:_texAccum[prevIdx] atIndex:2];
    [encoder setTexture:_texMotion atIndex:3];
    [encoder setTexture:_texDepth[currIdx] atIndex:4];
    [encoder setTexture:_texDepth[prevIdx] atIndex:5];
    [encoder setTexture:_texDirectLight atIndex:6];

    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroup];
    [encoder popDebugGroup];

    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:7 withBarrier:YES];
    
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 4: Spatial Denoising
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:8 withBarrier:NO];

    [encoder pushDebugGroup:@"Pass 4: Denoise"];
    [encoder setComputePipelineState:_psoDenoise];

    [encoder setTexture:_texDenoised atIndex:0]; // Output
    [encoder setTexture:_texAccum[currIdx] atIndex:1]; // Input: Current Accum
    [encoder setTexture:_texNormal atIndex:2];
    [encoder setTexture:_texDepth[currIdx] atIndex:3];

    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroup];
    [encoder popDebugGroup];

    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:9 withBarrier:YES];
    
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // -----------------------------------------------------------
    // PASS 5: Composite
    // -----------------------------------------------------------
    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:10 withBarrier:NO];

    [encoder pushDebugGroup:@"Pass 5: Composite"];
    [encoder setComputePipelineState:_psoComposite];

    [encoder setTexture:_texFinal atIndex:0]; // Final Output
    [encoder setTexture:_texDirectLight atIndex:1];
    [encoder setTexture:_texDenoised atIndex:2];
    [encoder setTexture:_texAlbedo atIndex:3];
    [encoder setTexture:_texDepth[currIdx] atIndex:4];


    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroup];
    [encoder popDebugGroup];

    if (_supportsTimestamps) [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:11 withBarrier:YES];

    _frameIndex++;
    const_cast<Character&>(character).lastRenderedViewProjectionMatrix = character.unjitteredViewProjectionMatrix;
}


void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called directly.");
}

id MetalRenderer::GetOutputTexture()
{
    return _texFinal;
}
