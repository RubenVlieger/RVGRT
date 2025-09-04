#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

#include "cumath.cuh"

// The CuTex class has been updated to use a linear device pointer (float*)
// for memory allocation, which is more flexible for writing directly from
// a kernel. It then creates a texture object from this linear memory
// for efficient reading in other kernels.
class CuTex {
public:
    // Main constructor to create and manage the linear buffer and texture object.
    // It assumes a single-channel float format (F32).
    CuTex(int _width, int _height,
          cudaChannelFormatDesc _cudaChannelFormatDesc,
          cudaTextureAddressMode _cudaTextureAddressMode,
          cudaTextureFilterMode _cudaTextureFilterMode);

    // Default constructor.
    CuTex();

    // Destructor to free allocated resources.
    ~CuTex();

    // Read back into CPU buffer. For a float texture, reinterpret as float*.
    // The buffer size must be width * height * sizeof(float).
    void readback(float* buffer);

    // Getters for kernel calls.
    // Returns the texture object for reading.
    cudaTextureObject_t getTexObj() const { return tex_obj; }
    // Returns the raw device pointer for writing.
    float* getDevPtr() const { return dev_ptr; }

    // Test whether the texture was successfully created.
    bool valid() const { return tex_obj != 0 && dev_ptr != nullptr; }

private:
    cudaTextureObject_t tex_obj = 0;
    float* dev_ptr = nullptr;
    int width = 0, height = 0;
};
