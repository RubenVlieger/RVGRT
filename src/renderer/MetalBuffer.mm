#import <Metal/Metal.h>
#include "renderer/MetalBuffer.hpp"
#include "cumath.h" 
#include <stdexcept>

MetalBuffer::MetalBuffer(id metalDevice)
{
    if (!metalDevice) {
        throw std::runtime_error("MetalBuffer received a null MTLDevice.");
    }
    m_device = metalDevice;
    // Retain the device to ensure it stays alive as long as the buffer does.
    [m_device retain];
    m_buffer = nil;
}

MetalBuffer::~MetalBuffer()
{
    Free();
    [m_device release];
}

void MetalBuffer::Allocate(uint64_t sizeInBytes)
{
    if (m_buffer) Free();

    // For simplicity and to support Readback, we use Shared storage mode.
    // For performance, Private would be better, but would require a separate
    // staging buffer and a blit command encoder to read back.
    MTLResourceOptions options = MTLResourceStorageModeShared;
    
    m_buffer = [m_device newBufferWithLength:sizeInBytes options:options];
    if (!m_buffer) {
        // In a real app, you might throw or handle this more gracefully.
        NSLog(@"FATAL: Failed to allocate MTLBuffer with size %llu", sizeInBytes);
        abort();
    }
}

void MetalBuffer::Free()
{
    if (m_buffer) {
        [m_buffer release];
        m_buffer = nil;
    }
}

uint64_t MetalBuffer::GetSize() const
{
    if (m_buffer) {
        return [(id<MTLBuffer>)m_buffer length];
    }
    return 0;
}

void MetalBuffer::Readback(void* cpuBuffer, uint64_t sizeInBytes)
{
    if (!m_buffer || !cpuBuffer) return;
    
    uint64_t copySize = MIN(sizeInBytes, [(id<MTLBuffer>)m_buffer length]);
    
    // Because we created the buffer with Shared memory, we can directly
    // access its contents with a simple memcpy.
    memcpy(cpuBuffer, [(id<MTLBuffer>)m_buffer contents], copySize);
}

void* MetalBuffer::GetNativeHandle() const
{
    return m_buffer;
}