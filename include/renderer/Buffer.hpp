#pragma once
#include <cstdint>

// Forward declare the Objective-C `id` type only when compiling Objective-C++
// This allows the header to be included by pure C++ files without error.
#ifdef __OBJC__
@protocol MTLBuffer;
#else
typedef void* id;
#endif

// A pure abstract base class for a generic GPU buffer.
class Buffer
{
public:
    virtual ~Buffer() = default;

    // Allocates a specific number of bytes on the GPU.
    virtual void Allocate(uint64_t sizeInBytes) = 0;

    // Releases the GPU memory.
    virtual void Free() = 0;

    // Returns the size of the allocated buffer in bytes.
    virtual uint64_t GetSize() const = 0;

    // Copies data from the GPU buffer back to a CPU-side pointer.
    // The cpuBuffer must be pre-allocated with at least sizeInBytes.
    virtual void Readback(void* cpuBuffer, uint64_t sizeInBytes) = 0;

    // Returns the native, platform-specific handle to the GPU resource.
    // - On CUDA, this will be a `uint32_t*` (or other device pointer).
    // - On Metal, this will be an `id<MTLBuffer>`.
    // The caller is responsible for casting the void* to the correct type.
    virtual void* GetNativeHandle() const = 0;
};