#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__) && defined(__OBJC__)
#import <Metal/Metal.h>
#endif

namespace BrickPoolImpl {

#if defined(__APPLE__)

struct MetalBrickPoolTraits {
    using OccupancyBuffer = id;
    using DataBuffer = id;
    using DeviceHandle = id;

    static DeviceHandle GetDevice();
    
    static OccupancyBuffer AllocateOccupancy(DeviceHandle device, size_t size);
    static DataBuffer AllocateData(DeviceHandle device, size_t size);
    static void ZeroOccupancy(DeviceHandle device, OccupancyBuffer buffer, size_t size);
    static void ZeroData(DeviceHandle device, DataBuffer buffer, size_t size);
    
    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
};

using PlatformBrickPoolTraits = MetalBrickPoolTraits;

#elif defined(_WIN32)

struct CudaBrickPoolTraits {
    using OccupancyBuffer = uint64_t*;
    using DataBuffer = uint8_t*;
    using DeviceHandle = void*;

    static DeviceHandle GetDevice();
    
    static OccupancyBuffer AllocateOccupancy(DeviceHandle device, size_t size);
    static DataBuffer AllocateData(DeviceHandle device, size_t size);
    static void ZeroOccupancy(DeviceHandle device, OccupancyBuffer buffer, size_t size);
    static void ZeroData(DeviceHandle device, DataBuffer buffer, size_t size);
    static void FreeBuffer(void* ptr);
    
    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
};

using PlatformBrickPoolTraits = CudaBrickPoolTraits;

#else

// Forward-declare WebGPU buffer type
struct WGPUBufferImpl;
typedef struct WGPUBufferImpl* WGPUBuffer;

struct WebBrickPoolTraits {
    using OccupancyBuffer = WGPUBuffer;
    using DataBuffer = WGPUBuffer;      // Host uploads packed u32 data for WGSL
    using DeviceHandle = void*;         // WebGraphicsDevice pointer

    static DeviceHandle GetDevice();
    
    static OccupancyBuffer AllocateOccupancy(DeviceHandle device, size_t size);
    static DataBuffer AllocateData(DeviceHandle device, size_t size);
    static void ZeroOccupancy(DeviceHandle device, OccupancyBuffer buffer, size_t size);
    static void ZeroData(DeviceHandle device, DataBuffer buffer, size_t size);
    static void FreeBuffer(WGPUBuffer buffer);
    
    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
};

using PlatformBrickPoolTraits = WebBrickPoolTraits;

#endif

}
