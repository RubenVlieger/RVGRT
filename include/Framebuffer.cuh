#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>


class Framebuffer {
public:
    // Constructors & Destructor
    Framebuffer();
    ~Framebuffer();

    // Allocate GPU memory
    void Allocate(int width, int height);

    // Free GPU memory
    void Free();

    // Read back GPU buffer to CPU
    std::vector<uint32_t> readback() const;
    void readback(uint32_t* buffer) const;


    // Get GPU pointer (for kernel rendering)
    uint32_t* devicePtr() const { return d_pixels; }

    // Get dimensions
    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    int width;
    int height;
    uint32_t* d_pixels; // GPU pointer
};