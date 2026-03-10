#include "renderer/CUDA/CudaBuffer.cuh"
#include "cumath.h"
#include "TerrainGeneration.h"

// The world generation kernel remains the same.
extern "C" __global__
void fillKernel(uint32_t* __restrict__ data, uint64_t numWords)
{
    // ... (same exact kernel logic as before) ...
    uint64_t wordIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (wordIdx >= numWords) return;

    uint64_t baseBit = wordIdx * 32ull;
    uint32_t w = 0u;
    for (uint64_t bit = 0; bit < 32; ++bit) {
        uint64_t bitIndex = baseBit + bit;
        uint64_t z = bitIndex >> (SHIX + SHIY);
        uint64_t y = (bitIndex >> SHIX) & (uint64_t)MODY;
        uint64_t x = bitIndex & (uint64_t)MODX;
        float v = Evaluate((float)x, (float)y, (float)z);
        if (v > 0.7f) w |= (1u << bit);
    }
    data[wordIdx] = w;
}


// --- CudaBuffer Method Implementations ---

CudaBuffer::CudaBuffer() {}
CudaBuffer::~CudaBuffer()
{
    Free();
}

void CudaBuffer::Allocate(uint64_t sizeInBytes)
{
    if (m_deviceData) Free();
    m_sizeInBytes = sizeInBytes;
    CUDA_CHECK(cudaMalloc(&m_deviceData, m_sizeInBytes));
}

void CudaBuffer::Free()
{
    if (m_deviceData)
    {
        cudaFree(m_deviceData);
        m_deviceData = nullptr;
        m_sizeInBytes = 0;
    }
}

uint64_t CudaBuffer::GetSize() const
{
    return m_sizeInBytes;
}

void CudaBuffer::Readback(void* cpuBuffer, uint64_t sizeInBytes)
{
    if (!m_deviceData || !cpuBuffer) return;
    uint64_t copySize = sizeInBytes < m_sizeInBytes ? sizeInBytes : m_sizeInBytes;
    CUDA_CHECK(cudaMemcpy(cpuBuffer, m_deviceData, copySize, cudaMemcpyDeviceToHost));
}

void* CudaBuffer::GetNativeHandle() const
{
    return m_deviceData;
}

void CudaBuffer::FillWithWorldData()
{
    if (!m_deviceData) return;
    
    uint64_t numWords = m_sizeInBytes / sizeof(uint32_t);
    const unsigned int threads = 256;
    unsigned int blocks = (unsigned int)((numWords + threads - 1) / threads);

    fillKernel<<<blocks, threads>>>(m_deviceData, numWords);
    CUDA_CHECK(cudaGetLastError());
}
