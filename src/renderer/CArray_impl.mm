#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "CArray.h"
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/MetalDevice.hpp"

namespace{ 
id<MTLDevice> get_metal_device()
{
    GraphicsDevice* gDevice = State::state.graphicsDevice.get();
    if (!gDevice) {
        throw std::runtime_error("CArray Error: GraphicsDevice is not initialized.");
    }
    MetalDevice* mDevice = static_cast<MetalDevice*>(gDevice);
    return mDevice->GetMetalDevice();
}
}


void CArray::Allocate(uint64_t _size)
{
    if (dev_data) { Free(); }

    SIZE = _size;
    id<MTLDevice> device = get_metal_device();

    MTLResourceOptions options = MTLResourceStorageModePrivate;

    id<MTLBuffer> buffer = [device newBufferWithLength:SIZE options:options];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate private MTLBuffer in CArray.");
    }

    dev_data = reinterpret_cast<uint32_t*>((__bridge_retained void*)buffer);
    std::cout << "Private Metal CArray allocated with size: " << SIZE << " bytes." << std::endl;
}

void CArray::Free()
{
    if (dev_data)
    {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)reinterpret_cast<void*>(dev_data);
        buffer = nil;
        dev_data = nullptr;
        SIZE = 0;
    }
}

void CArray::fill()
{
    if (!dev_data) {
        throw std::runtime_error("CArray::fill() called before Allocate().");
    }

    id<MTLDevice> device = get_metal_device();

    NSError *error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
        throw std::runtime_error("Could not load default Metal library. Check build settings.");
    }
    id<MTLFunction> kernelFunc = [library newFunctionWithName:@"CArray_fill_kernel"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
    if (!pso) {
         throw std::runtime_error("Failed to create compute pipeline state for CArray_fill_kernel.");
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pso];
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(dev_data);
    [encoder setBuffer:buffer offset:0 atIndex:0];

    uint64_t numWords = SIZE / sizeof(uint32_t);
    MTLSize gridSize = MTLSizeMake(numWords, 1, 1);
    NSUInteger threadGroupSize = [pso maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > numWords) { threadGroupSize = numWords; }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted]; 
    std::cout << "CArray::fill() Metal kernel finished." << std::endl;
}

void CArray::readback(uint32_t* cpu_buffer)
{
    if (!dev_data || !cpu_buffer) {
        throw std::runtime_error("Invalid buffer for CArray::readback().");
    }

    std::cout << "PERFORMANCE WARNING: Executing CArray::readback(). This is a slow, synchronous operation intended for debugging only." << std::endl;

    id<MTLDevice> device = get_metal_device();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLBuffer> privateBuffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(dev_data);

    id<MTLBuffer> stagingBuffer = [device newBufferWithLength:SIZE options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder];

    [blitEncoder copyFromBuffer:privateBuffer
                   sourceOffset:0
                       toBuffer:stagingBuffer
              destinationOffset:0
                           size:SIZE];
    [blitEncoder endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    memcpy(cpu_buffer, [stagingBuffer contents], SIZE);
}
