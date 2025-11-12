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

    // --- CUDA-Specific Functionality ---
    // This is not part of the interface, as it's unique to the CUDA path.
    // It launches the terrain generation kernel.
    void FillWithWorldData();

private:
    uint32_t* m_deviceData = nullptr;
    uint64_t m_sizeInBytes = 0;
};