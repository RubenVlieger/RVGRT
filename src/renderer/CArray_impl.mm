#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "CArray.h"
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/MetalDevice.hpp"

static id<MTLDevice> get_metal_device() 
{
    GraphicsDevice* gDevice = State::state.graphicsDevice.get();
    if (!gDevice) {
        throw std::runtime_error("GraphicsDevice is not initialized.");
    }
    MetalDevice* mDevice = static_cast<MetalDevice*>(gDevice);
    return mDevice->GetMetalDevice();
}

void CArray::Allocate(uint64_t _size)
{
    if (dev_data) { // If a buffer already exists, free it first.
        Free();
    }
    
    SIZE = _size;
    id<MTLDevice> device = get_metal_device();

    // Use Shared storage mode so the CPU can access the buffer's contents for `readback`.
    MTLResourceOptions options = MTLResourceStorageModeShared;

    id<MTLBuffer> buffer = [device newBufferWithLength:SIZE options:options];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate MTLBuffer in CArray.");
    }

    // Retain the buffer and store it in the dev_data pointer via a cast.
    dev_data = reinterpret_cast<uint32_t*>((__bridge_retained void*)buffer);
    std::cout << "CREATING METAL ARRAY" << std::endl;
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
        std::cerr << "ERROR: CArray not allocated before fill()" << std::endl;
        return;
    }

    id<MTLDevice> device = get_metal_device();
    
    // Get the Kernel from the Library
    NSError *error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
        throw std::runtime_error("Could not load default Metal library.");
    }
    id<MTLFunction> kernelFunc = [library newFunctionWithName:@"CArray_fill_kernel"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
    if (!pso) {
         throw std::runtime_error("Failed to create compute pipeline state for CArray_fill_kernel.");
    }

    // Dispatch the Kernel 
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pso];
    
    // Get the buffer back from the pointer and set it as an argument for the kernel.
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(dev_data);
    [encoder setBuffer:buffer offset:0 atIndex:0];

    // Calculate grid and threadgroup sizes.
    uint64_t numWords = SIZE / sizeof(uint32_t);
    MTLSize gridSize = MTLSizeMake(numWords, 1, 1);
    
    NSUInteger threadGroupSize = [pso maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > numWords) {
        threadGroupSize = numWords;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Dispatch threads and commit.
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted]; // Block until the GPU is finished generating the world.
}


void CArray::readback(uint32_t* buffer)
{
    if (!dev_data || !buffer) {
        std::cerr << "ERROR: No device buffer or destination CPU buffer provided for readback." << std::endl;
        return;
    }

    // Cast the pointer back to a Metal buffer.
    id<MTLBuffer> metalBuffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(dev_data);
    
    // Because the buffer was created with MTLResourceStorageModeShared,
    // we can directly access its memory with memcpy.
    memcpy(buffer, [metalBuffer contents], SIZE);
}