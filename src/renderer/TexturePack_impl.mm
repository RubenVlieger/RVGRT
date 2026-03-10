#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "Texturepack.h"
#include "texturepackdata.h" // The embedded PNG data
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/Metal/MetalDevice.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {
id<MTLDevice> get_metal_device() {
    GraphicsDevice* gDevice = State::state.graphicsDevice.get();
    if (!gDevice) throw std::runtime_error("Texturepack Error: GraphicsDevice not initialized.");
    return static_cast<MetalDevice*>(gDevice)->GetMetalDevice();
}
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
    const int BLOCK_SIZE = 16;
    int blocksX = w / BLOCK_SIZE;
    int blocksY = h / BLOCK_SIZE;
    int totalLayers = blocksX * blocksY;

    id<MTLDevice> device = get_metal_device();

    // 1. Create Texture Array
    MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    descriptor.width = BLOCK_SIZE;
    descriptor.height = BLOCK_SIZE;
    descriptor.arrayLength = totalLayers;
    // Calculate mip levels: log2(16) + 1 = 5
    descriptor.mipmapLevelCount = (NSUInteger)(floor(log2((double)BLOCK_SIZE))) + 1;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) throw std::runtime_error("Failed to create Texture Array.");
    [texture setLabel:@"VoxelTextureArray"];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];

    // 2. Create ONE Big Staging Buffer for the entire atlas
    // This fixes the race condition. The data is constant and won't be overwritten.
    NSUInteger totalBytes = w * h * 4;
    id<MTLBuffer> stagingBuffer = [device newBufferWithBytes:rgba8 
                                                      length:totalBytes 
                                                     options:MTLResourceStorageModeShared];

    // 3. Loop through layers and blit from the big buffer
    int layerIndex = 0;
    NSUInteger bytesPerRowInAtlas = w * 4; // Stride of the full atlas image

    for (int by = 0; by < blocksY; ++by) {
        for (int bx = 0; bx < blocksX; ++bx) {
            
            // Calculate where this block starts in the big linear buffer
            // Offset = (Row * Width + Col) * BytesPerPixel
            NSUInteger sourceOffset = (by * BLOCK_SIZE * w + bx * BLOCK_SIZE) * 4;

            [blit copyFromBuffer:stagingBuffer
                    sourceOffset:sourceOffset
               sourceBytesPerRow:bytesPerRowInAtlas // Key: Tells Metal to skip the rest of the atlas row
             sourceBytesPerImage:0
                      sourceSize:MTLSizeMake(BLOCK_SIZE, BLOCK_SIZE, 1)
                       toTexture:texture
                destinationSlice:layerIndex
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];

            layerIndex++;
        }
    }

    [blit generateMipmapsForTexture:texture];
    [blit endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    texObj_ = (__bridge_retained void*)texture;
    std::cout << "Texture Array uploaded correctly (Race condition fixed)." << std::endl;
}