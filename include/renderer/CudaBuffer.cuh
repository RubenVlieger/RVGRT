#pragma once
#include "renderer/Buffer.hpp"
#include <cstdint>

// The CUDA implementation of the Buffer interface.
class CudaBuffer : public Buffer
{
public:
    CudaBuffer();
    ~CudaBuffer() override;

    // --- Interface Implementation ---
    void Allocate(uint64_t sizeInBytes) override;
    void Free() override;
    uint64_t GetSize() const override;
    void Readback(void* cpuBuffer, uint64_t sizeInBytes) override;
    void* GetNativeHandle() const override;

private:
    void* m_deviceData = nullptr;
    uint64_t m_sizeInBytes = 0;
};