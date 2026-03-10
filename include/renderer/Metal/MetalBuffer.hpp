#pragma once
#include "renderer/Buffer.hpp"

// The Metal implementation of the Buffer interface.
class MetalBuffer : public Buffer
{
public:
    // Requires the Metal device to perform allocations.
    MetalBuffer(id metalDevice);
    ~MetalBuffer() override;

    // --- Interface Implementation ---
    void Allocate(uint64_t sizeInBytes) override;
    void Free() override;
    uint64_t GetSize() const override;
    void Readback(void* cpuBuffer, uint64_t sizeInBytes) override;
    void* GetNativeHandle() const override;

private:
    id m_device; // id<MTLDevice>
    id m_buffer; // id<MTLBuffer>
};
