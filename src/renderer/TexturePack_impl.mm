#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "Texturepack.h"
#include "texturepackdata.h"      // The embedded PNG data
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/MetalDevice.hpp"

// STB Image: A single-file image loading library.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Helper to get the global Metal device
static id<MTLDevice> get_metal_device() {
    GraphicsDevice* gDevice = State::state.graphicsDevice.get();
    if (!gDevice) throw std::runtime_error("GraphicsDevice not initialized.");
    return static_cast<MetalDevice*>(gDevice)->GetMetalDevice();
}


// --- Metal/Objective-C++ Implementation of Texturepack ---

Texturepack::Texturepack()
{
    int nChannels = 0;
    // Load the embedded PNG from texturepack.h into a raw byte buffer.
    unsigned char* image_data = stbi_load_from_memory(
        texturepack_png, (int)texturepack_png_len, &width_, &height_, &nChannels, 4
    );
    if (!image_data) {
        throw std::runtime_error(std::string("stbi_load_from_memory failed: ") + stbi_failure_reason());
    }

    // Upload the raw data to a Metal texture.
    uploadRGBAFloat(image_data, width_, height_);

    stbi_image_free(image_data);
    std::cout << "Texturepack loaded into MTLTexture." << std::endl;
}

Texturepack::~Texturepack() {
    releaseResources();
}

void Texturepack::releaseResources() {
    if (texObj_) {
        // Cast the stored void* back to the Metal texture type and release it.
        id<MTLTexture> texture = (__bridge id<MTLTexture>)texObj_;
        [texture release];
        texObj_ = nullptr;
    }
}

void Texturepack::uploadRGBAFloat(const unsigned char* rgba8, int w, int h)
{
    id<MTLDevice> device = get_metal_device();

    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm // Use 8-bit RGBA
                                   width:w
                                  height:h
                               mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead; // We only need to read from it in shaders.

    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) {
        throw std::runtime_error("Failed to create MTLTexture.");
    }

    // Define the region of the texture we want to copy data into (in this case, the whole thing).
    MTLRegion region = MTLRegionMake2D(0, 0, w, h);

    // Copy the pixel data from the CPU buffer to the GPU texture.
    NSUInteger bytesPerRow = 4 * w; // 4 channels (R,G,B,A) * width
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:rgba8
               bytesPerRow:bytesPerRow];
    
    // Store the Metal texture in our class member.
    // We use a __bridge_retained cast to transfer ownership to our C++ class.
    texObj_ = (__bridge_retained void*)texture;
}