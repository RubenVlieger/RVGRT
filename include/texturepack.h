#pragma once


#ifndef __METAL_VERSION__
#include "cumath.h"
#include <utility> // for std::exchange


class Texturepack {
public:
    Texturepack(const unsigned char* pngData, size_t pngSize);
    Texturepack();
    ~Texturepack();

    // no copy semantics for simplicity
    Texturepack(const Texturepack&) = delete;
    Texturepack& operator=(const Texturepack&) = delete;

    // Move constructor
    Texturepack(Texturepack&& other) noexcept
        : texObj_(std::exchange(other.texObj_, nullptr)),
          cuArray_(std::exchange(other.cuArray_, nullptr)),
          width_(std::exchange(other.width_, 0)),
          height_(std::exchange(other.height_, 0)) {}

    // Move assignment operator
    Texturepack& operator=(Texturepack&& other) noexcept {
        if (this != &other) {
            // Free current resources
            releaseResources();
            
            // Transfer ownership from other
            texObj_ = std::exchange(other.texObj_, nullptr);
            cuArray_ = std::exchange(other.cuArray_, nullptr);
            width_ = std::exchange(other.width_, 0);
            height_ = std::exchange(other.height_, 0);
        }
        return *this;
    }

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

private:
    void uploadRGBAFloat(const unsigned char* rgba8, int w, int h);
    void releaseResources(); 
    TEXTURE_OBJECT texObj_ = nullptr;
    ARRAY_OBJECT cuArray_ = nullptr;
    int width_ = 0, height_ = 0;
};
#endif