#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cumath.cuh"
#include <utility> // for std::exchange

// Simple half3 struct wrapper

class Texturepack {
public:
    // Construct from a PNG blob in memory (eg the embedded header produced by xxd -i)
    // pngData: pointer to PNG bytes, pngSize: length in bytes
    Texturepack(const unsigned char* pngData, size_t pngSize);
    Texturepack();
    ~Texturepack();

    // no copy semantics for simplicity
    Texturepack(const Texturepack&) = delete;
    Texturepack& operator=(const Texturepack&) = delete;

    // Move constructor
    Texturepack(Texturepack&& other) noexcept
        : texObj_(std::exchange(other.texObj_, 0)),
          cuArray_(std::exchange(other.cuArray_, nullptr)),
          width_(std::exchange(other.width_, 0)),
          height_(std::exchange(other.height_, 0)) {}

    // Move assignment operator
    Texturepack& operator=(Texturepack&& other) noexcept {
        if (this != &other) {
            // Free current resources
            releaseResources();
            
            // Transfer ownership from other
            texObj_ = std::exchange(other.texObj_, 0);
            cuArray_ = std::exchange(other.cuArray_, nullptr);
            width_ = std::exchange(other.width_, 0);
            height_ = std::exchange(other.height_, 0);
        }
        return *this;
    }

    // Accessors
    cudaTextureObject_t texObject() const { return texObj_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Device sampling helpers (callable from kernels)
    // u,v in [0,1], normalizedCoords = true in the texture object
    __device__ static float3 sampleFloat3(cudaTextureObject_t tex, float u, float v);

    // Swap function for efficient swapping
    void swap(Texturepack& other) noexcept {
        std::swap(texObj_, other.texObj_);
        std::swap(cuArray_, other.cuArray_);
        std::swap(width_, other.width_);
        std::swap(height_, other.height_);
    }

private:
    void uploadRGBAFloat(const unsigned char* rgba8, int w, int h);
    void releaseResources(); // Helper to release CUDA resources

    cudaTextureObject_t texObj_ = 0;
    cudaArray_t        cuArray_ = nullptr;
    int width_ = 0, height_ = 0;
};