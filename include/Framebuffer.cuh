#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#define NOMINMAX
#include "windowsx.h"
#include "d3d11.h"

class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();

    void Allocate(int width, int height);
    void Free();

    // ==== New for D3D11 interop ====
    void InitializeInterop(ID3D11Device* device, ID3D11DeviceContext* context);
    void CleanupInterop();
    void CopyDeviceToTexture();  // maps resource and copies d_pixels into D3D texture

    // (Existing: readback or devicePtr if needed)
    std::vector<uint32_t> readback() const;
    void readback(uint32_t* buffer) const;

    uint32_t* devicePtr() const { return d_pixels; }
    ID3D11Texture2D* getD3DTexture() const { return d_texture; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    int width, height;
    uint32_t* d_pixels = nullptr;
    cudaGraphicsResource* cudaResource = nullptr;
    ID3D11Texture2D* d_texture = nullptr;
    ID3D11DeviceContext* d_context = nullptr;
};
