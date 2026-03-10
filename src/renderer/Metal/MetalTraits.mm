#include "renderer/RendererTraits.hpp"

#if defined(__APPLE__)

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <cstdarg>
#include <cstring>

namespace RendererImpl {

// ============================================================================
// Metal Renderer Traits Implementation
// ============================================================================

MetalRendererTraits::Device MetalRendererTraits::GetDevice() {
    // This would typically be obtained from the platform layer
    // For now, return nullptr - the actual device is passed to the constructor
    return nullptr;
}

MetalRendererTraits::RenderTarget MetalRendererTraits::CreateRenderTarget(
    Device device, 
    uint32_t width, 
    uint32_t height, 
    TextureFormat format,
    const char* name) {
    
    RenderTarget target;
    target.width = width;
    target.height = height;
    target.format = format;
    
    id<MTLDevice> dev = (id<MTLDevice>)device;
    
    // Convert TextureFormat to MTLPixelFormat
    MTLPixelFormat pixelFormat;
    switch (format) {
        case TextureFormat::RGBA16Float:
            pixelFormat = MTLPixelFormatRGBA16Float;
            break;
        case TextureFormat::RGBA8Unorm:
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case TextureFormat::RGBA8Snorm:
            pixelFormat = MTLPixelFormatRGBA8Snorm;
            break;
        case TextureFormat::RG16Float:
            pixelFormat = MTLPixelFormatRG16Float;
            break;
        case TextureFormat::R32Float:
            pixelFormat = MTLPixelFormatR32Float;
            break;
        case TextureFormat::R32Uint:
            pixelFormat = MTLPixelFormatR32Uint;
            break;
        default:
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
    }
    
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    
    id<MTLTexture> texture = [dev newTextureWithDescriptor:desc];
    if (name) {
        texture.label = [NSString stringWithUTF8String:name];
    }
    
    target.texture = texture;
    return target;
}



void MetalRendererTraits::DestroyRenderTarget(Device device, RenderTarget& target) {
    (void)device;
    if (target.texture) {
        // ARC handles release automatically
        target.texture = nullptr;
    }
    target.width = 0;
    target.height = 0;
}

MetalRendererTraits::Buffer MetalRendererTraits::CreateBuffer(Device device, size_t size, const char* name) {
    id<MTLDevice> dev = (id<MTLDevice>)device;
    id<MTLBuffer> buffer = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (name) {
        buffer.label = [NSString stringWithUTF8String:name];
    }
    return buffer;
}

void MetalRendererTraits::DestroyBuffer(Device device, Buffer buffer) {
    (void)device;
    (void)buffer;
    // ARC handles release automatically
}

void* MetalRendererTraits::MapBuffer(Buffer buffer) {
    id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
    return [buf contents];
}

void MetalRendererTraits::UnmapBuffer(Buffer buffer) {
    (void)buffer;
    // No-op for shared storage mode
}

void MetalRendererTraits::UploadBuffer(Buffer buffer, const void* data, size_t size) {
    id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
    memcpy([buf contents], data, size);
}

MetalRendererTraits::ComputeEncoder MetalRendererTraits::CreateEncoder(CommandBuffer cmdBuf, const char* name) {
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmdBuf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (name) {
        encoder.label = [NSString stringWithUTF8String:name];
    }
    return encoder;
}

void MetalRendererTraits::DestroyEncoder(ComputeEncoder encoder) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    [enc endEncoding];
}

void MetalRendererTraits::SetPipelineState(ComputeEncoder encoder, PipelineState pso) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)pso;
    [enc setComputePipelineState:pipeline];
}

void MetalRendererTraits::SetTexture(ComputeEncoder encoder, uint32_t index, Texture texture) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    id<MTLTexture> tex = (id<MTLTexture>)texture;
    [enc setTexture:tex atIndex:index];
}

void MetalRendererTraits::SetBuffer(ComputeEncoder encoder, uint32_t index, Buffer buffer, uint64_t offset) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
    [enc setBuffer:buf offset:offset atIndex:index];
}

void MetalRendererTraits::SetConstantData(ComputeEncoder encoder, uint32_t index, const void* data, size_t size) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    [enc setBytes:data length:size atIndex:index];
}

void MetalRendererTraits::DispatchKernel(ComputeEncoder encoder, GridSize grid, GroupSize group) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    MTLSize gridSize = MTLSizeMake(grid.width, grid.height, grid.depth);
    MTLSize threadGroupSize = MTLSizeMake(group.width, group.height, group.depth);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
}

void MetalRendererTraits::MemoryBarrier(ComputeEncoder encoder) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
    [enc memoryBarrierWithScope:MTLBarrierScopeTextures];
}

void MetalRendererTraits::ApplyUpscaling(CommandBuffer cmdBuf, Scaler scaler, 
                                         Texture input, Texture output,
                                         Texture depth, Texture motion,
                                         bool reset, float jitterX, float jitterY) {
    id<MTLFXTemporalScaler> temporalScaler = (id<MTLFXTemporalScaler>)scaler;
    if (!temporalScaler) return;
    
    // Set up scaler inputs
    temporalScaler.colorTexture = (id<MTLTexture>)input;
    temporalScaler.outputTexture = (id<MTLTexture>)output;
    temporalScaler.depthTexture = (id<MTLTexture>)depth;
    temporalScaler.motionTexture = (id<MTLTexture>)motion;
    temporalScaler.reset = reset;
    
    // Motion vector scale (matches the original implementation)
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmdBuf;
    
    // Encode upscaling to command buffer
    [temporalScaler encodeToCommandBuffer:cmd];
}

void MetalRendererTraits::Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    NSString* message = [[NSString alloc] initWithFormat:[NSString stringWithUTF8String:format] arguments:args];
    va_end(args);
    NSLog(@"%@", message);
}

void MetalRendererTraits::LogError(const char* format, ...) {
    va_list args;
    va_start(args, format);
    NSString* message = [[NSString alloc] initWithFormat:[NSString stringWithUTF8String:format] arguments:args];
    va_end(args);
    NSLog(@"ERROR: %@", message);
}

} // namespace RendererImpl

#endif // __APPLE__
