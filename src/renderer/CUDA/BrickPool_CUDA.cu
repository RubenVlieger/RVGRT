#include "renderer/BrickPoolTraits.hpp"
#include <cuda_runtime.h>
#include <cstdarg>
#include <cstdio>

namespace BrickPoolImpl {

CudaBrickPoolTraits::DeviceHandle CudaBrickPoolTraits::GetDevice() {
    return nullptr;
}

CudaBrickPoolTraits::OccupancyBuffer CudaBrickPoolTraits::AllocateOccupancy(DeviceHandle device, size_t size) {
    uint64_t* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBrickPool] ERROR: Failed to allocate occupancy buffer: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

CudaBrickPoolTraits::DataBuffer CudaBrickPoolTraits::AllocateData(DeviceHandle device, size_t size) {
    uint8_t* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBrickPool] ERROR: Failed to allocate data buffer: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void CudaBrickPoolTraits::ZeroOccupancy(DeviceHandle device, OccupancyBuffer buffer, size_t size) {
    cudaError_t err = cudaMemset(buffer, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBrickPool] ERROR: Failed to zero occupancy buffer: %s\n", cudaGetErrorString(err));
    }
}

void CudaBrickPoolTraits::ZeroData(DeviceHandle device, DataBuffer buffer, size_t size) {
    cudaError_t err = cudaMemset(buffer, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBrickPool] ERROR: Failed to zero data buffer: %s\n", cudaGetErrorString(err));
    }
}

void CudaBrickPoolTraits::FreeBuffer(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void CudaBrickPoolTraits::Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("[BrickPool] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void CudaBrickPoolTraits::LogError(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[CudaBrickPool] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

}
