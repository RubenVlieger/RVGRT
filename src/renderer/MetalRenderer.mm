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

MetalRenderer::MetalRenderer(id device_id) : _texturepack() 
{
    _device = (id<MTLDevice>)device_id;

    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];

    if (!defaultLibrary) {
        NSLog(@"FATAL ERROR: Failed to find the default shader library. Make sure your .metal files are included in the CMake target.");
        abort();
    }
    NSError *error = nil;

    id<MTLFunction> worldGenFunc = [defaultLibrary newFunctionWithName:@"GeneratePackedWorld"];
    if (!worldGenFunc) {
        NSLog(@"Error: Could not find function 'GeneratePackedWorld' in default library.");
    }
    _worldGenerationPSO = [_device newComputePipelineStateWithFunction:worldGenFunc error:&error];
    if (!_worldGenerationPSO) {
        NSLog(@"Error creating World Generation PSO: %@", error);
    }

   id<MTLFunction> approxFunc = [defaultLibrary newFunctionWithName:@"distApproximationKernel"];
    _distApproxPSO = [_device newComputePipelineStateWithFunction:approxFunc error:&error];
    if (!_distApproxPSO) NSLog(@"Error Pass 0: %@", error);

    // Pass 1: G-Buffer
    id<MTLFunction> gbufferFunc = [defaultLibrary newFunctionWithName:@"gbuffer_kernel"];
    _gBufferPSO = [_device newComputePipelineStateWithFunction:gbufferFunc error:&error];
    if (!_gBufferPSO) NSLog(@"Error Pass 1: %@", error);

    // Pass 2: Shadows
    id<MTLFunction> shadowFunc = [defaultLibrary newFunctionWithName:@"shadow_kernel"];
    _shadowPSO = [_device newComputePipelineStateWithFunction:shadowFunc error:&error];
    if (!_shadowPSO) NSLog(@"Error Pass 2: %@", error);

    // Pass 3: Reflections
    id<MTLFunction> reflFunc = [defaultLibrary newFunctionWithName:@"reflection_kernel"];
    _reflectionPSO = [_device newComputePipelineStateWithFunction:reflFunc error:&error];
    if (!_reflectionPSO) NSLog(@"Error Pass 3: %@", error);

    // Pass 4: Final Shading
    id<MTLFunction> shadingFunc = [defaultLibrary newFunctionWithName:@"shading_kernel"];
    _shadingPSO = [_device newComputePipelineStateWithFunction:shadingFunc error:&error];
    if (!_shadingPSO) NSLog(@"Error Pass 4: %@", error);
    auto t1 = std::chrono::high_resolution_clock::now();

    const int packedWidth  = SIZEX / 4;
    const int packedHeight = SIZEY / 4;
    const int packedDepth  = SIZEZ / 2;

    MTLTextureDescriptor *voxelDesc = [[MTLTextureDescriptor alloc] init];
    voxelDesc.textureType = MTLTextureType3D;

    voxelDesc.pixelFormat = MTLPixelFormatR32Uint; 
    voxelDesc.width = packedWidth;
    voxelDesc.height = packedHeight;
    voxelDesc.depth = packedDepth;
    voxelDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    voxelDesc.storageMode = MTLStorageModePrivate;

    _voxelTexture = [_device newTextureWithDescriptor:voxelDesc];
    if(!_voxelTexture) {
        NSLog(@"Failed to create Bit-Packed Voxel Texture");
    }

    id<MTLCommandQueue> queue = [_device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:_worldGenerationPSO]; 
    [encoder setTexture:_voxelTexture atIndex:0];

    MTLSize threadGroupSize = MTLSizeMake(8, 8, 8);
    MTLSize gridSize = MTLSizeMake(packedWidth, packedHeight, packedDepth);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    auto t2 = std::chrono::high_resolution_clock::now();

    NSLog(@"Voxel world generated in %.2f ms", std::chrono::duration<double, std::milli>(t2 - t1).count());


    _csdf.AllocateSDF();
    _csdf.GenerateSDF(_voxelTexture);
    auto t3 = std::chrono::high_resolution_clock::now();
    NSLog(@"CSDF generated in %.2f ms", std::chrono::duration<double, std::milli>(t3 - t2).count());


   _giData.AllocateGI();
    _giData.InitializeGIData(_voxelTexture, _csdf, _texturepack);
    auto t4 = std::chrono::high_resolution_clock::now();
    NSLog(@"GI grid initialized in %.2f ms", std::chrono::duration<double, std::milli>(t4 - t3).count());

    createRenderTarget(State::dispWIDTH, State::dispHEIGHT);
}


MetalRenderer::~MetalRenderer() {}

void MetalRenderer::createRenderTarget(uint32_t width, uint32_t height) 
{
    MTLTextureDescriptor *mainDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    mainDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    mainDesc.storageMode = MTLStorageModePrivate;
    _renderTargetTexture = [_device newTextureWithDescriptor:mainDesc];

    MTLTextureDescriptor *posDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:NO];
    posDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    posDesc.storageMode = MTLStorageModePrivate;
    _gBufferPosTexture = [_device newTextureWithDescriptor:posDesc];

    MTLTextureDescriptor *normDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:width height:height mipmapped:NO];
    normDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    normDesc.storageMode = MTLStorageModePrivate;
    _gBufferNormTexture = [_device newTextureWithDescriptor:normDesc];

    MTLTextureDescriptor *shadowDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:width height:height mipmapped:NO];
    shadowDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    shadowDesc.storageMode = MTLStorageModePrivate;
    _shadowMaskTexture = [_device newTextureWithDescriptor:shadowDesc];

    MTLTextureDescriptor *reflDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:width height:height mipmapped:NO];
    reflDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    reflDesc.storageMode = MTLStorageModePrivate;
    _reflectionTexture = [_device newTextureWithDescriptor:reflDesc];

    uint32_t halfW = width / 2;
    uint32_t halfH = height / 2;
    
    MTLTextureDescriptor *distDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:halfW height:halfH mipmapped:NO];
    distDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    distDesc.storageMode = MTLStorageModePrivate;
    _halfDistTexture = [_device newTextureWithDescriptor:distDesc];
}

void MetalRenderer::OnResize(uint32_t newWidth, uint32_t newHeight)
{
    if ([(id<MTLTexture>)_renderTargetTexture width] != newWidth || [(id<MTLTexture>)_renderTargetTexture height] != newHeight) {
        createRenderTarget(newWidth, newHeight);
    }
}

void MetalRenderer::Draw(id<MTLComputeCommandEncoder> encoder, const Character& character, unsigned int frameCount)
{
    // --- Setup Data ---
    CameraData camData;
    camData.position = { (float)character.position.x, (float)character.position.y, (float)character.position.z };
    camData.forward  = { (float)character.direction.x, (float)character.direction.y, (float)character.direction.z };
    camData.right    = { character.camera.right.x, character.camera.right.y, character.camera.right.z };
    camData.up       = { character.camera.up.x, character.camera.up.y, character.camera.up.z };

    FrameData frameData;
    frameData.sunDirection = simd_normalize(simd_make_float3(10.f, 5.f, -4.f));
    frameData.time = (float)CFAbsoluteTimeGetCurrent();

    MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
    
    // Grids
    MTLSize gridHalf = MTLSizeMake([(id<MTLTexture>)_halfDistTexture width], [(id<MTLTexture>)_halfDistTexture height], 1);
    MTLSize gridFull = MTLSizeMake([(id<MTLTexture>)_renderTargetTexture width], [(id<MTLTexture>)_renderTargetTexture height], 1);

    // PASS 0: DISTANCE APPROX (Helper for G-Buffer)
    [encoder pushDebugGroup:@"Pass 0: Approx"];
    [encoder setComputePipelineState:_distApproxPSO];
    [encoder setTexture:_halfDistTexture atIndex:0];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    [encoder dispatchThreads:gridHalf threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
    [encoder popDebugGroup];

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // PASS 1: G-BUFFER (Visibility)
    [encoder pushDebugGroup:@"Pass 1: G-Buffer"];
    [encoder setComputePipelineState:_gBufferPSO];
    [encoder setTexture:_gBufferPosTexture atIndex:0];
    [encoder setTexture:_gBufferNormTexture atIndex:1];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    [encoder setTexture:_halfDistTexture atIndex:6];
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroupSize];
    [encoder popDebugGroup];

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // PASS 2: SHADOW MASK
    [encoder pushDebugGroup:@"Pass 2: Shadows"];
    [encoder setComputePipelineState:_shadowPSO];
    [encoder setTexture:_shadowMaskTexture atIndex:0];
    [encoder setTexture:_gBufferPosTexture atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:0];
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroupSize];
    [encoder popDebugGroup];

    // PASS 3: REFLECTIONS
    [encoder pushDebugGroup:@"Pass 3: Reflections"];
    [encoder setComputePipelineState:_reflectionPSO];
    [encoder setTexture:_reflectionTexture atIndex:0];
    [encoder setTexture:_gBufferPosTexture atIndex:1]; 
    [encoder setTexture:_gBufferNormTexture atIndex:2];
    [encoder setTexture:_voxelTexture atIndex:3];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:4];
    [encoder setTexture:(id<MTLTexture>)_texturepack.getTextureObject() atIndex:5];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroupSize];
    [encoder popDebugGroup];

    // Barrier: Wait for Shadows and Reflections to finish writing
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // PASS 4: FINAL SHADING 
    [encoder pushDebugGroup:@"Pass 4: Compose"];
    [encoder setComputePipelineState:_shadingPSO];
    [encoder setTexture:_renderTargetTexture atIndex:0];    // Out: Screen
    [encoder setTexture:_gBufferPosTexture atIndex:1];      // In: Pos
    [encoder setTexture:_gBufferNormTexture atIndex:2];     // In: Norm
    [encoder setTexture:_shadowMaskTexture atIndex:3];      // In: Shadow Mask
    [encoder setTexture:_reflectionTexture atIndex:4];      // In: Reflection
    [encoder setTexture:(id<MTLTexture>)_texturepack.getTextureObject() atIndex:5]; // In: Atlas
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroupSize];
    [encoder popDebugGroup];
}

void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called. This path is not intended for the main render loop.");
}


id MetalRenderer::GetOutputTexture()
{
    return _renderTargetTexture;
}