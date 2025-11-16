#import <MetalKit/MetalKit.h>
#include "renderer/MetalRenderer.hpp"
#include "State.hpp"
#include "Character.hpp"
#include "renderer/ShaderTypes.h"

#include "renderer/MetalDevice.hpp"
#include "renderer/MetalBuffer.hpp"
#include "cumath.h"


MetalRenderer::MetalRenderer(id device_id)
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

    GenerateWorld();


    createRenderTarget(State::dispWIDTH, State::dispHEIGHT);
}

void MetalRenderer::GenerateWorld()
{
    NSLog(@"Allocating and generating voxel world on GPU via CArray...");
    _voxelBuffer.Allocate(BYTESIZE);
    
    _voxelBuffer.fill();    
    NSLog(@"World generation complete.");

}

MetalRenderer::~MetalRenderer() {}

void MetalRenderer::createRenderTarget(uint32_t width, uint32_t height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    _renderTargetTexture = [_device newTextureWithDescriptor:descriptor];
}

void MetalRenderer::OnResize(uint32_t newWidth, uint32_t newHeight)
{
    if ([(id<MTLTexture>)_renderTargetTexture width] != newWidth || [(id<MTLTexture>)_renderTargetTexture height] != newHeight) {
        createRenderTarget(newWidth, newHeight);
    }
}

void MetalRenderer::Draw(const Character& character, unsigned int frameCount)
{
    MetalDevice* metalDevice = static_cast<MetalDevice*>(State::state.graphicsDevice.get());
    id<MTLCommandQueue> commandQueue = metalDevice->GetMetalCommandQueue();

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    commandBuffer.label = @"Main Frame Command Buffer";

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"RayTraceEncoder";
    
    [encoder setComputePipelineState:_computePSO];
    [encoder setTexture:_renderTargetTexture atIndex:0];

    CameraData camData;
    camData.position = { (float)character.position.x, (float)character.position.y, (float)character.position.z };
    camData.forward  = { (float)character.direction.x, (float)character.direction.y, (float)character.direction.z };
    camData.right    = { character.camera.right.x, character.camera.right.y, character.camera.right.z };
    camData.up       = { character.camera.up.x, character.camera.up.y, character.camera.up.z };

    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];

    
    id<MTLBuffer> nativeBuffer = (id<MTLBuffer>)_voxelBuffer.getPtr();
    [encoder setBuffer:nativeBuffer offset:0 atIndex:1];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    MTLSize threadgridSize = MTLSizeMake([(id<MTLTexture>)_renderTargetTexture width], [(id<MTLTexture>)_renderTargetTexture height], 1);
    
    [encoder dispatchThreads:threadgridSize threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
    
    [commandBuffer commit];
    
    [commandBuffer waitUntilCompleted];
}

id MetalRenderer::GetOutputTexture()
{
    return _renderTargetTexture;
}