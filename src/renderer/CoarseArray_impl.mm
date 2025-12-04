#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "CoarseArray.h"
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/MetalDevice.hpp"


namespace{ 
    id<MTLDevice> get_metal_device() {
        GraphicsDevice* gDevice = State::state.graphicsDevice.get();
        if (!gDevice) throw std::runtime_error("GraphicsDevice not initialized.");
        return static_cast<MetalDevice*>(gDevice)->GetMetalDevice();
    }
}

static id<MTLComputePipelineState> create_pso(id<MTLDevice> device, const char* kernel_name) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) throw std::runtime_error("Could not load default Metal library.");
    
    id<MTLFunction> function = [library newFunctionWithName:@(kernel_name)];
    if (!function) {
        throw std::runtime_error(std::string("Failed to find kernel function: ") + kernel_name);
    }
    
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pso) {
        throw std::runtime_error(std::string("Failed to create PSO for ") + kernel_name +
                                 ". Error: " + [error.localizedDescription UTF8String]);
    }
    return pso;
}


CoarseArray::CoarseArray() {
    id<MTLDevice> device = get_metal_device();
    _psoDistX = create_pso(device, "CoarseArray_computeDistX");
    _psoDistY = create_pso(device, "CoarseArray_computeDistY");
    _psoDistZ = create_pso(device, "CoarseArray_computeDistZ");
}

CoarseArray::~CoarseArray() { 
    _sdfTexture = nullptr;
}

// Helper to create 3D textures
id<MTLTexture> create3DTexture(id<MTLDevice> device, NSUInteger width, NSUInteger height, NSUInteger depth, MTLPixelFormat format) {
    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = format;
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:desc];
}

void CoarseArray::AllocateSDF() {
    id<MTLDevice> device = get_metal_device();
    _sdfTexture = (__bridge_retained void*)create3DTexture(device, SDF_SIZEX, SDF_SIZEY, SDF_SIZEZ, MTLPixelFormatR16Float);
    std::cout << "Allocated SDF 3D Texture" << std::endl;
}


void* CoarseArray::getSDFTexture() { return _sdfTexture; }

unsigned char* CoarseArray::getPtr() { return nullptr; }

void CoarseArray::GenerateSDF(void* packedVoxelTexture) {
    id<MTLDevice> device = get_metal_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    
    // We need a temporary texture for the ping-pong passes
    id<MTLTexture> tempTex = create3DTexture(device, SDF_SIZEX, SDF_SIZEY, SDF_SIZEZ, MTLPixelFormatR16Float);
    id<MTLTexture> sdfTex = (__bridge id<MTLTexture>)_sdfTexture;

    MTLSize gridSize = MTLSizeMake(SDF_SIZEX, SDF_SIZEY, SDF_SIZEZ);
    MTLSize threadgroupSize = MTLSizeMake(8, 8, 8);
    
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    
    // Pass 1: X (Read Voxel -> Write Temp)
    [enc setComputePipelineState:(id<MTLComputePipelineState>)_psoDistX];
    [enc setTexture:(__bridge id<MTLTexture>)packedVoxelTexture atIndex:0];
    [enc setTexture:tempTex atIndex:1];
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    // Pass 2: Y (Read Temp -> Write SDF)
    [enc setComputePipelineState:(id<MTLComputePipelineState>)_psoDistY];
    [enc setTexture:tempTex atIndex:0];
    [enc setTexture:sdfTex atIndex:1];
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    // Pass 3: Z (Read SDF -> Write Temp)
    [enc setComputePipelineState:(id<MTLComputePipelineState>)_psoDistZ];
    [enc setTexture:sdfTex atIndex:0];
    [enc setTexture:tempTex atIndex:1];
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [enc endEncoding];
    
    // Blit Temp back to SDF (final result)
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit copyFromTexture:tempTex 
              sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) 
               sourceSize:MTLSizeMake(SDF_SIZEX, SDF_SIZEY, SDF_SIZEZ)
                toTexture:sdfTex 
         destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
         
    [blit endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}
