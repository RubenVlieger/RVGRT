#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// ============================================================================
// Forward declarations for platform-specific types
// ============================================================================

// Metal types (Objective-C)
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
@protocol MTLComputePipelineState;
@protocol MTLTexture;
@protocol MTLBuffer;
@protocol MTLCommandQueue;
@protocol MTLFXTemporalScaler;
#else
typedef void* id;
#endif

// CUDA types (forward declared to avoid including cuda_runtime.h in headers)
struct cudaArray;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef struct CUstream_st* cudaStream_t;

// ============================================================================
// Texture format enumeration (platform-agnostic)
// ============================================================================

enum class TextureFormat {
    RGBA16Float,    // Half-precision RGBA
    RGBA8Unorm,     // 8-bit unsigned normalized RGBA
    RGBA8Snorm,     // 8-bit signed normalized RGBA
    RG16Float,      // Half-precision RG
    R32Float,       // Single-channel float
    R32Uint,        // Single-channel unsigned int
};

// ============================================================================
// Grid configuration for kernel dispatch
// ============================================================================

struct GridSize {
    uint32_t width;
    uint32_t height;
    uint32_t depth = 1;
};

struct GroupSize {
    uint32_t width;
    uint32_t height;
    uint32_t depth = 1;
};

// ============================================================================
// Render Target Descriptor
// ============================================================================

struct RenderTargetDesc {
    uint32_t width;
    uint32_t height;
    TextureFormat format;
    const char* name;  // For debugging
    bool isHalfRes = false;  // For half-resolution targets like volumetric
};

// ============================================================================
// Pipeline State Handles (opaque types)
// ============================================================================

struct PipelineStateHandle {
#ifdef __APPLE__
    id metalPSO = nullptr;
#else
    void* cudaKernel = nullptr;
#endif
};

// ============================================================================
// Renderer Traits Namespace
// ============================================================================

namespace RendererImpl {

// ============================================================================
// Metal Renderer Traits
// ============================================================================
#if defined(__APPLE__)

struct MetalRendererTraits {
    // Type aliases
    using Device = id;
    using CommandBuffer = id;
    using ComputeEncoder = id;
    using PipelineState = id;
    using Texture = id;
    using Buffer = id;
    using Scaler = id;
    using CommandQueue = id;
    
    // Render target structure
    struct RenderTarget {
        Texture texture = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        TextureFormat format;
    };
    
    // Material map type (forward declared - actual type depends on platform)
    using MaterialMapType = class MaterialMap;
    
    // Static methods (implemented in MetalTraits.cpp)
    static Device GetDevice();
    
    static RenderTarget CreateRenderTarget(Device device, uint32_t width, uint32_t height, 
                                           TextureFormat format, const char* name);
    static void DestroyRenderTarget(Device device, RenderTarget& target);
    
    static Buffer CreateBuffer(Device device, size_t size, const char* name);
    static void DestroyBuffer(Device device, Buffer buffer);
    static void* MapBuffer(Buffer buffer);
    static void UnmapBuffer(Buffer buffer);
    static void UploadBuffer(Buffer buffer, const void* data, size_t size);
    
    static ComputeEncoder CreateEncoder(CommandBuffer cmdBuf, const char* name);
    static void DestroyEncoder(ComputeEncoder encoder);
    
    static void SetPipelineState(ComputeEncoder encoder, PipelineState pso);
    static void SetTexture(ComputeEncoder encoder, uint32_t index, Texture texture);
    static void SetBuffer(ComputeEncoder encoder, uint32_t index, Buffer buffer, uint64_t offset = 0);
    static void SetConstantData(ComputeEncoder encoder, uint32_t index, const void* data, size_t size);
    static void DispatchKernel(ComputeEncoder encoder, GridSize grid, GroupSize group);
    static void MemoryBarrier(ComputeEncoder encoder);
    
    static void ApplyUpscaling(CommandBuffer cmdBuf, Scaler scaler, Texture input, Texture output,
                               Texture depth, Texture motion, bool reset, float jitterX, float jitterY);
    
    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
};

using PlatformRendererTraits = MetalRendererTraits;

// ============================================================================
// CUDA Renderer Traits
// ============================================================================
#elif defined(_WIN32)

struct CudaRendererTraits {
    // Type aliases
    using Device = void*;  // Not used for CUDA, but kept for consistency
    using CommandBuffer = cudaStream_t;
    using ComputeEncoder = void*;  // CUDA doesn't use encoders
    using PipelineState = void*;  // Function pointers
    using Texture = cudaSurfaceObject_t;
    using Buffer = void*;
    using Scaler = void*;  // DLSS handle
    using CommandQueue = void*;  // Not used in CUDA
    
    // Render target structure (CUDA needs array + surface + texture)
    struct RenderTarget {
        cudaArray* array = nullptr;
        cudaSurfaceObject_t surface = 0;
        cudaTextureObject_t texture = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        TextureFormat format;
    };
    
    // Material map type (will be abstracted later)
    using MaterialMapType = class CudaMaterialMap;
    
    // Static methods (implemented in CudaTraits.cpp)
    static Device GetDevice();
    
    static RenderTarget CreateRenderTarget(Device device, uint32_t width, uint32_t height, 
                                           TextureFormat format, const char* name);
    static void DestroyRenderTarget(Device device, RenderTarget& target);
    
    static Buffer CreateBuffer(Device device, size_t size, const char* name);
    static void DestroyBuffer(Device device, Buffer buffer);
    static void* MapBuffer(Buffer buffer);
    static void UnmapBuffer(Buffer buffer);
    static void UploadBuffer(Buffer buffer, const void* data, size_t size);
    
    static ComputeEncoder CreateEncoder(CommandBuffer cmdBuf, const char* name);
    static void DestroyEncoder(ComputeEncoder encoder);
    
    static void SetPipelineState(ComputeEncoder encoder, PipelineState pso);
    static void SetTexture(ComputeEncoder encoder, uint32_t index, Texture texture);
    static void SetBuffer(ComputeEncoder encoder, uint32_t index, Buffer buffer, uint64_t offset = 0);
    static void SetConstantData(ComputeEncoder encoder, uint32_t index, const void* data, size_t size);
    static void DispatchKernel(ComputeEncoder encoder, PipelineState pso, GridSize grid, GroupSize group);
    static void MemoryBarrier(ComputeEncoder encoder);
    
    static void ApplyUpscaling(CommandBuffer cmdBuf, Scaler scaler, Texture input, Texture output,
                               Texture depth, Texture motion, bool reset, float jitterX, float jitterY);
    
    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
};

using PlatformRendererTraits = CudaRendererTraits;

// ============================================================================
// Web/WebGPU Renderer Traits (Placeholder for future)
// ============================================================================
#else

struct WebRendererTraits {
    // Placeholder for WebGPU implementation
    using Device = void*;
    using CommandBuffer = void*;
    using ComputeEncoder = void*;
    using PipelineState = void*;
    using Texture = void*;
    using Buffer = void*;
    using Scaler = void*;
    using CommandQueue = void*;
    
    struct RenderTarget {
        Texture texture = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        TextureFormat format;
    };
    
    using MaterialMapType = void*;
    
    static Device GetDevice();
    static RenderTarget CreateRenderTarget(Device device, uint32_t width, uint32_t height, 
                                           TextureFormat format, const char* name);
    static void DestroyRenderTarget(Device device, RenderTarget& target);
    static Buffer CreateBuffer(Device device, size_t size, const char* name);
    static void DestroyBuffer(Device device, Buffer buffer);
    static void* MapBuffer(Buffer buffer);
    static void UnmapBuffer(Buffer buffer);
    static void UploadBuffer(Buffer buffer, const void* data, size_t size);
    static ComputeEncoder CreateEncoder(CommandBuffer cmdBuf, const char* name);
    static void DestroyEncoder(ComputeEncoder encoder);
    static void SetPipelineState(ComputeEncoder encoder, PipelineState pso);
    static void SetTexture(ComputeEncoder encoder, uint32_t index, Texture texture);
    static void SetBuffer(ComputeEncoder encoder, uint32_t index, Buffer buffer, uint64_t offset = 0);
    static void SetConstantData(ComputeEncoder encoder, uint32_t index, const void* data, size_t size);
    static void DispatchKernel(ComputeEncoder encoder, PipelineState pso, GridSize grid, GroupSize group);
    static void MemoryBarrier(ComputeEncoder encoder);
    static void ApplyUpscaling(CommandBuffer cmdBuf, Scaler scaler, Texture input, Texture output,
                               Texture depth, Texture motion, bool reset, float jitterX, float jitterY);
    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
};

using PlatformRendererTraits = WebRendererTraits;

#endif

} // namespace RendererImpl

// ============================================================================
// Convenience alias for current platform
// ============================================================================

using CurrentRendererTraits = RendererImpl::PlatformRendererTraits;
