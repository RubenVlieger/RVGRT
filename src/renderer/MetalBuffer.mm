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
    m_buffer = nil;
}

MetalBuffer::~MetalBuffer()
{
    Free();
}

void MetalBuffer::Allocate(uint64_t sizeInBytes)
{
    if (m_buffer) Free();

    MTLResourceOptions options = MTLResourceStorageModeShared;
    
    m_buffer = [m_device newBufferWithLength:sizeInBytes options:options];
    if (!m_buffer) {
        NSLog(@"FATAL: Failed to allocate MTLBuffer with size %llu", sizeInBytes);
        abort();
    }
}

void MetalBuffer::Free()
{
    if (m_buffer) {
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
    
    memcpy(cpuBuffer, [(id<MTLBuffer>)m_buffer contents], copySize);
}

void* MetalBuffer::GetNativeHandle() const
{
    return (__bridge void*)m_buffer;
}
