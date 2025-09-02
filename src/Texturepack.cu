// Texturepack.cu
#define NOMINMAX  // Prevent windows.h from defining min/max macros

#include "Texturepack.cuh"
#include "texturepack.h"      // <-- include embedded PNG
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>
// stb_image (single-file decoder)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


// Helper conversion
static inline __half float_to_half(float f) { return __float2half_rn(f); }

// ------------------------------
// Constructor: loads from embedded PNG by default
Texturepack::Texturepack()
{
    int nChannels = 0;
    // stbi loads rows top->bottom. We request 4 channels (RGBA) for simplicity.
    unsigned char* image = stbi_load_from_memory(texturepack_png, (int)texturepack_png_len, &width_, &height_, &nChannels, 4);
    if (!image) {
        throw std::runtime_error(std::string("stbi_load_from_memory failed: ") + stbi_failure_reason());
    }

    // Upload to CUDA array as float4 normalized [0,1]
    uploadRGBAFloat(image, width_, height_);

    stbi_image_free(image);
}

// Optional: allow custom PNGs
Texturepack::Texturepack(const unsigned char* pngData, size_t pngSize)
{
    if (!pngData || pngSize == 0) throw std::runtime_error("Texturepack: invalid png data");

    int nChannels = 0;
    unsigned char* image = stbi_load_from_memory(pngData, (int)pngSize, &width_, &height_, &nChannels, 4);
    if (!image) {
        throw std::runtime_error(std::string("stbi_load_from_memory failed: ") + stbi_failure_reason());
    }

    uploadRGBAFloat(image, width_, height_);

    stbi_image_free(image);
}

Texturepack::~Texturepack()
{
    if (texObj_) {
        cudaDestroyTextureObject(texObj_);
        texObj_ = 0;
    }
    if (cuArray_) {
        cudaFreeArray(cuArray_);
        cuArray_ = nullptr;
    }
}

void Texturepack::uploadRGBAFloat(const unsigned char* rgba8, int w, int h)
{
    // Convert 8-bit RGBA -> float4 (normalized 0..1)
    std::vector<float> tmp; // layout: pixel0.r,pixel0.g,pixel0.b,pixel0.a, pixel1...
    tmp.resize(size_t(w) * size_t(h) * 4);

    const unsigned char* src = rgba8;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x);
            tmp[size_t(idx)*4 + 0] = src[0] / 255.0f;
            tmp[size_t(idx)*4 + 1] = src[1] / 255.0f;
            tmp[size_t(idx)*4 + 2] = src[2] / 255.0f;
            tmp[size_t(idx)*4 + 3] = src[3] / 255.0f;
            src += 4;
        }
    }

    // Create CUDA array (float4 channels)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaError_t err = cudaMallocArray(&cuArray_, &channelDesc, w, h);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocArray failed: ") + cudaGetErrorString(err));
    }

    size_t spitch = size_t(w) * sizeof(float) * 4; // bytes per row
    err = cudaMemcpy2DToArray(
        cuArray_, 0, 0, tmp.data(), spitch, spitch, size_t(h), cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        cudaFreeArray(cuArray_);
        cuArray_ = nullptr;
        throw std::runtime_error(std::string("cudaMemcpy2DToArray failed: ") + cudaGetErrorString(err));
    }

    // Create resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray_;

    // Create texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    err = cudaCreateTextureObject(&texObj_, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        cudaFreeArray(cuArray_);
        cuArray_ = nullptr;
        texObj_ = 0;
        throw std::runtime_error(std::string("cudaCreateTextureObject failed: ") + cudaGetErrorString(err));
    }
}

// Device helpers
__device__ float3 Texturepack::sampleFloat3(cudaTextureObject_t tex, float u, float v)
{
    float4 t = tex2D<float4>(tex, u, v);
    return make_float3(t.x, t.y, t.z);
}
