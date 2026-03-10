#ifdef _WIN32
#include "renderer/CudaBuffer.cuh"
#include <cuda_runtime.h>
#include <cstdio>

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBuffer] %s: %s\n", msg, cudaGetErrorString(err));
    }
}

CudaBuffer::CudaBuffer() {}
CudaBuffer::~CudaBuffer() { Free(); }

void CudaBuffer::Allocate(uint64_t sizeInBytes) {
    if (m_deviceData) Free();
    m_sizeInBytes = sizeInBytes;
    checkCuda(cudaMalloc(&m_deviceData, sizeInBytes), "Allocate");
}

void CudaBuffer::Free() {
    if (m_deviceData) {
        cudaFree(m_deviceData);
        m_deviceData = nullptr;
        m_sizeInBytes = 0;
    }
}

uint64_t CudaBuffer::GetSize() const { return m_sizeInBytes; }

void CudaBuffer::Readback(void* cpuBuffer, uint64_t sizeInBytes) {
    if (!m_deviceData || !cpuBuffer) return;
    uint64_t copySize = sizeInBytes < m_sizeInBytes ? sizeInBytes : m_sizeInBytes;
    checkCuda(cudaMemcpy(cpuBuffer, m_deviceData, copySize, cudaMemcpyDeviceToHost), "Readback");
}

void* CudaBuffer::GetNativeHandle() const { return m_deviceData; }
#endif