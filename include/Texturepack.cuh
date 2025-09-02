#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cumath.cuh"

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

    // Accessors
    cudaTextureObject_t texObject() const { return texObj_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Device sampling helpers (callable from kernels)
    // u,v in [0,1], normalizedCoords = true in the texture object
    __device__ static float3 sampleFloat3(cudaTextureObject_t tex, float u, float v);

private:
    void uploadRGBAFloat(const unsigned char* rgba8, int w, int h);

    cudaTextureObject_t texObj_ = 0;
    cudaArray_t        cuArray_ = nullptr;
    int width_ = 0, height_ = 0;
};