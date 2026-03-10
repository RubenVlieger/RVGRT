#include "renderer/RendererTraits.hpp"

#if defined(_WIN32)

#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace RendererImpl {

// ============================================================================
// CUDA Error Checking Helper
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// CUDA Renderer Traits Implementation
// ============================================================================

CudaRendererTraits::Device CudaRendererTraits::GetDevice() {
    // CUDA doesn't use a device handle in the same way as Metal
    // The device is implicit through the CUDA context
    return nullptr;
}

CudaRendererTraits::RenderTarget CudaRendererTraits::CreateRenderTarget(
    Device device, 
    uint32_t width, 
    uint32_t height, 
    TextureFormat format,
    const char* name) {
    
    (void)device;
    (void)name;
    
    RenderTarget target;
    target.width = width;
    target.height = height;
    target.format = format;
    target.array = nullptr;
    target.surface = 0;
    target.texture = 0;
    
    // Create CUDA channel format descriptor
    cudaChannelFormatDesc channelDesc;
    switch (format) {
        case TextureFormat::RGBA16Float:
            channelDesc = cudaCreateChannelDescHalf4();
            break;
        case TextureFormat::RGBA8Unorm:
            channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            break;
        case TextureFormat::RGBA8Snorm:
            channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned);
            break;
        case TextureFormat::RG16Float:
            channelDesc = cudaCreateChannelDescHalf2();
            break;
        case TextureFormat::R32Float:
            channelDesc = cudaCreateChannelDesc<float>();
            break;
        case TextureFormat::R32Uint:
            channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
            break;
        default:
            channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            break;
    }
    
    // Allocate CUDA array
    CUDA_CHECK(cudaMallocArray(&target.array, &channelDesc, width, height, cudaArraySurfaceLoadStore));
    
    // Create surface object (for writing)
    cudaResourceDesc surfRes = {};
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = target.array;
    CUDA_CHECK(cudaCreateSurfaceObject(&target.surface, &surfRes));
    
    // Create texture object (for reading with filtering)
    cudaResourceDesc texRes = {};
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = target.array;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // false - pixel coordinates
    
    CUDA_CHECK(cudaCreateTextureObject(&target.texture, &texRes, &texDesc, nullptr));
    
    return target;
}

void CudaRendererTraits::DestroyRenderTarget(Device device, RenderTarget& target) {
    (void)device;
    
    if (target.texture) {
        cudaDestroyTextureObject(target.texture);
        target.texture = 0;
    }
    if (target.surface) {
        cudaDestroySurfaceObject(target.surface);
        target.surface = 0;
    }
    if (target.array) {
        cudaFreeArray(target.array);
        target.array = nullptr;
    }
    
    target.width = 0;
    target.height = 0;
}

CudaRendererTraits::Buffer CudaRendererTraits::CreateBuffer(Device device, size_t size, const char* name) {
    (void)device;
    (void)name;
    
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void CudaRendererTraits::DestroyBuffer(Device device, Buffer buffer) {
    (void)device;
    if (buffer) {
        cudaFree(buffer);
    }
}

void* CudaRendererTraits::MapBuffer(Buffer buffer) {
    // CUDA doesn't have a map/unmap concept like Metal
    // For simplicity, we allocate host memory and sync manually
    // A more sophisticated implementation would use cudaHostAlloc/cudaMemcpy
    (void)buffer;
    return nullptr;
}

void CudaRendererTraits::UnmapBuffer(Buffer buffer) {
    (void)buffer;
}

void CudaRendererTraits::UploadBuffer(Buffer buffer, const void* data, size_t size) {
    CUDA_CHECK(cudaMemcpy(buffer, data, size, cudaMemcpyHostToDevice));
}

CudaRendererTraits::ComputeEncoder CudaRendererTraits::CreateEncoder(CommandBuffer cmdBuf, const char* name) {
    (void)name;
    // CUDA doesn't use encoders like Metal
    // The command buffer (stream) is used directly
    return cmdBuf;
}

void CudaRendererTraits::DestroyEncoder(ComputeEncoder encoder) {
    // No-op for CUDA
    (void)encoder;
}

void CudaRendererTraits::SetPipelineState(ComputeEncoder encoder, PipelineState pso) {
    // CUDA doesn't use pipeline states like Metal
    // Kernel functions are called directly
    (void)encoder;
    (void)pso;
}

void CudaRendererTraits::SetTexture(ComputeEncoder encoder, uint32_t index, Texture texture) {
    // CUDA kernels receive textures as arguments, not through encoder
    // This would be handled by the kernel argument setup
    (void)encoder;
    (void)index;
    (void)texture;
}

void CudaRendererTraits::SetBuffer(ComputeEncoder encoder, uint32_t index, Buffer buffer, uint64_t offset) {
    // CUDA kernels receive buffers as arguments
    (void)encoder;
    (void)index;
    (void)buffer;
    (void)offset;
}

void CudaRendererTraits::SetConstantData(ComputeEncoder encoder, uint32_t index, const void* data, size_t size) {
    // CUDA uses __constant__ memory or kernel arguments
    // This would be handled by the kernel argument setup
    (void)encoder;
    (void)index;
    (void)data;
    (void)size;
}

void CudaRendererTraits::DispatchKernel(ComputeEncoder encoder, PipelineState pso, GridSize grid, GroupSize group) {
    // For CUDA, we need to call the actual kernel function
    // This is platform-specific and would be implemented differently
    // The pso would be a function pointer to the kernel
    (void)encoder;
    (void)pso;
    (void)grid;
    (void)group;
}

void CudaRendererTraits::DispatchKernel(ComputeEncoder encoder, GridSize grid, GroupSize group) {
    (void)encoder;
    (void)grid;
    (void)group;
}

void CudaRendererTraits::MemoryBarrier(ComputeEncoder encoder) {
    // CUDA memory barrier (sync across all threads)
    (void)encoder;
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaRendererTraits::ApplyUpscaling(CommandBuffer cmdBuf, Scaler scaler, 
                                         Texture input, Texture output,
                                         Texture depth, Texture motion,
                                         bool reset, float jitterX, float jitterY) {
    // DLSS/Streamline SDK integration would go here
    // For now, this is a placeholder
    (void)cmdBuf;
    (void)scaler;
    (void)input;
    (void)output;
    (void)depth;
    (void)motion;
    (void)reset;
    (void)jitterX;
    (void)jitterY;
}

void CudaRendererTraits::Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void CudaRendererTraits::LogError(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
}

} // namespace RendererImpl

#endif // _WIN32
