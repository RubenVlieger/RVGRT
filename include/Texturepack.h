#pragma once


#ifndef __METAL_VERSION__
#include "cumath.h"
#include <utility> // for std::exchange


class Texturepack {
public:
    Texturepack(const unsigned char* pngData, size_t pngSize);
    Texturepack();
    ~Texturepack();

    Texturepack(const Texturepack&);
    Texturepack& operator=(const Texturepack&);

    Texturepack(Texturepack&& other) noexcept;
    Texturepack& operator=(Texturepack&& other) noexcept;

    // Accessors
    TEXTURE_OBJECT texObject() const { return texObj_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Device sampling helpers (platform-specific definitions)
    #if defined(__METAL_VERSION__)
    // For Metal, the TEXTURE_OBJECT is a texture2d, which is a value type.
    GPU_FUNC static float3 sampleFloat3(TEXTURE_OBJECT tex, sampler s, float u, float v);
    #elif defined(__CUDA_ARCH__)
    // For CUDA, TEXTURE_OBJECT is a cudaTextureObject_t (a handle).
    __device__ static float3 sampleFloat3(TEXTURE_OBJECT tex, float u, float v);
    #endif

    // Swap function for efficient swapping
    void swap(Texturepack& other) noexcept {
        std::swap(texObj_, other.texObj_);
        std::swap(cuArray_, other.cuArray_);
        std::swap(width_, other.width_);
        std::swap(height_, other.height_);
    }
    TEXTURE_OBJECT getTextureObject() const { return texObj_; }

private:
    void uploadRGBAData(const unsigned char* rgba8, int w, int h);
    void releaseResources(); 
    TEXTURE_OBJECT texObj_ = nullptr;
    ARRAY_OBJECT cuArray_ = nullptr;
    int width_ = 0, height_ = 0;
};
#endif