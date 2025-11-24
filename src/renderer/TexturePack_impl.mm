#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "Texturepack.h"
#include "texturepackdata.h" // The embedded PNG data
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/MetalDevice.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static id<MTLDevice> get_metal_device() {
    GraphicsDevice* gDevice = State::state.graphicsDevice.get();
    if (!gDevice) throw std::runtime_error("Texturepack Error: GraphicsDevice not initialized.");
    return static_cast<MetalDevice*>(gDevice)->GetMetalDevice();
}


Texturepack::Texturepack()
{
    int nChannels = 0;
    // Load the embedded PNG from texturepackdata.h into a raw byte buffer.
    // We request 4 channels (RGBA) to ensure consistent data layout.
    unsigned char* image_data = stbi_load_from_memory(
        texturepack_png, (int)texturepack_png_len, &width_, &height_, &nChannels, 4
    );
    if (!image_data) {
        throw std::runtime_error(std::string("stbi_load_from_memory failed: ") + stbi_failure_reason());
    }

    // Upload the raw pixel data to a Metal texture.
    uploadRGBAData(image_data, width_, height_);

    // Free the temporary CPU-side buffer.
    stbi_image_free(image_data);
}

Texturepack::~Texturepack() {
    releaseResources();
}

Texturepack::Texturepack(Texturepack&& other) noexcept
    : texObj_(std::exchange(other.texObj_, nullptr)),
      cuArray_(std::exchange(other.cuArray_, nullptr)), // cuArray is unused on Metal but we move it for consistency
      width_(std::exchange(other.width_, 0)),
      height_(std::exchange(other.height_, 0)) {}

Texturepack& Texturepack::operator=(Texturepack&& other) noexcept {
    if (this != &other) {
        releaseResources();
        texObj_ = std::exchange(other.texObj_, nullptr);
        cuArray_ = std::exchange(other.cuArray_, nullptr);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
    }
    return *this;
}

void Texturepack::releaseResources() {
    if (texObj_) {
        // Cast the stored void* back to the Metal texture type and transfer
        // ownership back to ARC to release it.
        id<MTLTexture> texture = (__bridge_transfer id<MTLTexture>)texObj_;
        texture = nil;
        texObj_ = nullptr;
    }
}

// Creates a Metal texture and uploads the pixel data to it.
void Texturepack::uploadRGBAData(const unsigned char* rgba8, int w, int h)
{
    id<MTLDevice> device = get_metal_device();

    // 1. Describe the texture we want to create.
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB // Use 8-bit sRGB for proper color.
                                   width:w
                                  height:h
                               mipmapped:NO]; // Mipmaps can be generated later if needed.

    // This texture will only be read from by shaders.
    // Making it private ensures it's stored in the fastest VRAM.
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;

    // 2. Create the GPU-private texture object.
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) {
        throw std::runtime_error("Failed to create MTLTFFexture.");
    }
    [texture setLabel:@"VoxelTextureAtlas"];

    // 3. Define the region of the texture to upload into (the whole thing).
    MTLRegion region = MTLRegionMake2D(0, 0, w, h);
    NSUInteger bytesPerRow = 4 * w; // 4 channels (R,G,B,A) * width.

    // 4. Upload the data. Since the texture is private, this is done via a command buffer.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];

    // Create a temporary shared buffer to hold the pixel data for the transfer.
    id<MTLBuffer> stagingBuffer = [device newBufferWithBytes:rgba8 length:bytesPerRow * h options:MTLResourceStorageModeShared];

    [blit copyFromBuffer:stagingBuffer
            sourceOffset:0
       sourceBytesPerRow:bytesPerRow
     sourceBytesPerImage:0
              sourceSize:MTLSizeMake(w, h, 1)
               toTexture:texture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];

    [blit endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted]; // Wait for upload to finish.

    // 5. Store the final Metal texture handle in our class member.
    // We use a __bridge_retained cast to transfer ownership to our C++ class.
    texObj_ = (__bridge_retained void*)texture;

    std::cout << "Texturepack uploaded to private MTLTexture." << std::endl;
}