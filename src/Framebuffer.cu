#include "Framebuffer.cuh"
#define NOMINMAX
#include <windowsx.h>
#include "d3d11.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cuda_d3d11_interop.h>


Framebuffer::Framebuffer()
    : width(0), height(0), d_pixels(nullptr) {
    
}   

Framebuffer::~Framebuffer() {

}

void Framebuffer::InitializeInterop(ID3D11Device* device, ID3D11DeviceContext* context) {
    d_context = context;
    // Create the D3D11 texture
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width; desc.Height = height;
    desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0; desc.MiscFlags = 0;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &d_texture);
    if (FAILED(hr)) {
        std::cerr << "CreateTexture2D failed\n";
    }
    // Register with CUDA (allows mapping)
    cudaGraphicsD3D11RegisterResource(&cudaResource, 
                                      (ID3D11Resource*)d_texture, 
                                      cudaGraphicsRegisterFlagsNone);
}

void Framebuffer::CleanupInterop() {
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }
    if (d_texture) {
        d_texture->Release();
        d_texture = nullptr;
    }
}

void Framebuffer::CopyDeviceToTexture() {
    if (!cudaResource) return;
    // Map the D3D resource for CUDA access
    cudaGraphicsMapResources(1, &cudaResource, 0);
    // Get CUDA array for the texture
    cudaArray* dstArray = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&dstArray, cudaResource, 0, 0);
    // Copy device memory into the array (2D copy, no host)
    size_t pitch = width * sizeof(uint32_t);
    cudaMemcpy2DToArray(dstArray,
                        0, 0,
                        d_pixels, pitch,
                        pitch, height,
                        cudaMemcpyDeviceToDevice);
    // Unmap so D3D can use it
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
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

