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

    id<MTLFunction> kernelFunction = [defaultLibrary newFunctionWithName:@"raytrace_kernel"];
    NSError *error = nil;
    _computePSO = [_device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!_computePSO) {
        NSLog(@"Failed to create compute pipeline state, error: %@", error);
    }


    id<MTLFunction> approxFunction = [defaultLibrary newFunctionWithName:@"distApproximationKernel"];
    _distApproxPSO = [_device newComputePipelineStateWithFunction:approxFunction error:&error];
    if (!_distApproxPSO) NSLog(@"Error creating approx PSO: %@", error);



    id<MTLFunction> worldGenerationKernel = [defaultLibrary newFunctionWithName:@"GeneratePackedWorld"];
    _worldGenerationPSO = [_device newComputePipelineStateWithFunction:worldGenerationKernel error:&error];
    if (!_worldGenerationPSO) NSLog(@"Error creating approx PSO: %@", error);


    auto t1 = std::chrono::high_resolution_clock::now();

    NSUInteger packedWidth  = SIZEX / 2;
    NSUInteger packedHeight = SIZEY / 2;
    NSUInteger packedDepth  = SIZEZ / 2;

    MTLTextureDescriptor *voxelDesc = [[MTLTextureDescriptor alloc] init];
    voxelDesc.textureType = MTLTextureType3D;
    voxelDesc.pixelFormat = MTLPixelFormatR8Uint;
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
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModePrivate; 
    _renderTargetTexture = [_device newTextureWithDescriptor:descriptor];


    uint32_t halfW = max(width / 2, 1u);
    uint32_t halfH = max(height / 2, 1u);

    MTLTextureDescriptor *distDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                        width:halfW
                                                                                        height:halfH
                                                                                        mipmapped:NO];
                                                                                        

    distDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    distDesc.storageMode = MTLStorageModePrivate; 
    _halfDistTexture = [_device newTextureWithDescriptor:distDesc];

    MTLTextureDescriptor *shadDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                        width:halfW
                                                                                       height:halfH
                                                                                    mipmapped:NO];

    shadDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    shadDesc.storageMode = MTLStorageModePrivate; 
    _halfShadowTexture = [_device newTextureWithDescriptor:shadDesc];
}

void MetalRenderer::OnResize(uint32_t newWidth, uint32_t newHeight)
{
    if ([(id<MTLTexture>)_renderTargetTexture width] != newWidth || [(id<MTLTexture>)_renderTargetTexture height] != newHeight) {
        createRenderTarget(newWidth, newHeight);
    }
}

void MetalRenderer::Draw(id<MTLComputeCommandEncoder> encoder, const Character& character, unsigned int frameCount)
{
    CameraData camData;
    camData.position = { (float)character.position.x, (float)character.position.y, (float)character.position.z };
    camData.forward  = { (float)character.direction.x, (float)character.direction.y, (float)character.direction.z };
    camData.right    = { character.camera.right.x, character.camera.right.y, character.camera.right.z };
    camData.up       = { character.camera.up.x, character.camera.up.y, character.camera.up.z };

    FrameData frameData;
    frameData.sunDirection = simd_normalize(simd_make_float3(10.f, 5.f, -4.f));
    frameData.time = (float)CFAbsoluteTimeGetCurrent();


    // --- APPROX PASS ---
    [encoder pushDebugGroup:@"Distance Approx Pass"];
    [encoder setComputePipelineState:_distApproxPSO];
    
    [encoder setTexture:_halfDistTexture atIndex:0];
    [encoder setTexture:_halfShadowTexture atIndex:1];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    
    MTLSize threadgroupSize = MTLSizeMake(8, 4, 1);
    MTLSize gridHalf = MTLSizeMake([(id<MTLTexture>)_halfDistTexture width], 
                                   [(id<MTLTexture>)_halfDistTexture height], 1);
    [encoder dispatchThreads:gridHalf threadsPerThreadgroup:threadgroupSize];
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
    [encoder popDebugGroup];


    // --- MAIN PASS ---
    [encoder pushDebugGroup:@"Main Raytrace Pass"];
    [encoder setComputePipelineState:_computePSO];
    
    [encoder setTexture:_renderTargetTexture atIndex:0];
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:2];
    
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    [encoder setTexture:(__bridge id<MTLTexture>)_giData.getGITexture() atIndex:4];
    
    [encoder setTexture:(id<MTLTexture>)_texturepack.getTextureObject() atIndex:5];
    [encoder setTexture:_halfDistTexture atIndex:6];
    [encoder setTexture:_halfShadowTexture atIndex:7];

    MTLSize gridFull = MTLSizeMake([(id<MTLTexture>)_renderTargetTexture width], 
                                   [(id<MTLTexture>)_renderTargetTexture height], 1);
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadgroupSize];
    [encoder popDebugGroup];
}



void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called. This path is not intended for the main render loop.");
}


id MetalRenderer::GetOutputTexture()
{
    return _renderTargetTexture;
}