#pragma once


#ifndef __METAL_VERSION__
#include "cumath.h"
#include <cuda_runtime.h>
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
    #if !defined(__METAL_VERSION__)
    cudaTextureObject_t texObject() const { return texObj_; }
    #else
    TEXTURE_OBJECT texObject() const { return texObj_; }
    #endif
    int width() const { return width_; }
    int height() const { return height_; }

    // Device sampling helpers (platform-specific definitions)
    #if defined(__METAL_VERSION__)
    // For Metal, the TEXTURE_OBJECT is a texture2d, which is a value type.
    GPU_FUNC static float3 sampleFloat3(TEXTURE_OBJECT tex, sampler s, float u, float v);
    #else
    // For CUDA
    __device__ static float3 sampleFloat3(cudaTextureObject_t tex, float u, float v);
    #endif

    // Swap function for efficient swapping
    void swap(Texturepack& other) noexcept {
        std::swap(texObj_, other.texObj_);
        std::swap(cuArray_, other.cuArray_);
        std::swap(width_, other.width_);
        std::swap(height_, other.height_);
    }
    #if !defined(__METAL_VERSION__)
    cudaTextureObject_t getTextureObject() const { return texObj_; }
    #else
    TEXTURE_OBJECT getTextureObject() const { return texObj_; }
    #endif

private:
    void uploadRGBAData(const unsigned char* rgba8, int w, int h);
    void releaseResources(); 
    #if defined(__METAL_VERSION__)
    TEXTURE_OBJECT texObj_ = nullptr;
    ARRAY_OBJECT cuArray_ = nullptr;
    #else
    cudaTextureObject_t texObj_ = 0;
    cudaArray_t cuArray_ = nullptr;
    #endif
    int width_ = 0, height_ = 0;
};
#endif