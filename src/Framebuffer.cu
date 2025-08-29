#include "Framebuffer.cuh"
#include <cuda_runtime.h>
#include <iostream>

Framebuffer::Framebuffer()
    : width(0), height(0), d_pixels(nullptr) {
    
}   

Framebuffer::~Framebuffer() {

}

void Framebuffer::Allocate(int w, int h) {
    if (d_pixels) {
        Free(); // free old buffer
    }

    width = w;
    height = h;

    size_t size = width * height * sizeof(uint32_t);
    cudaError_t err = cudaMalloc(&d_pixels, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        d_pixels = nullptr;
    } else {
        // Optionally clear framebuffer
        cudaMemset(d_pixels, 0, size);
    }
}

void Framebuffer::Free() {
    if (d_pixels) {
        cudaFree(d_pixels);
        d_pixels = nullptr;
    }
}

std::vector<uint32_t> Framebuffer::readback() const {
    std::vector<uint32_t> cpuBuffer(width * height);
    if (d_pixels) {
        cudaMemcpy(cpuBuffer.data(), d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    return cpuBuffer;
}

void Framebuffer::readback(uint32_t* buffer) const {
    if (d_pixels) {
        cudaMemcpy(buffer, d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
}

