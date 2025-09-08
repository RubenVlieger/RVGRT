#ifdef D3D12
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#define NOMINMAX
#include "windows.h"

// D3D12 specific headers
#include <d3d12.h>
#include <wrl/client.h> // For Microsoft::WRL::ComPtr

class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();

    void InitializeInterop(ID3D12Device* device, int width, int height);
    void CleanupInterop();
    
    // Getters
    ID3D12Resource* getD3DTexture() const { return d_texture.Get(); }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
    // NEW/REVISED Getters for the kernel
    uchar4* getDevicePtr() const { return d_pixels; }
    size_t getPitchInBytes() const { return rowPitchInBytes; }

private:
    int width = 0, height = 0;
    size_t rowPitchInBytes = 0; // The pitch of the D3D12 texture
    uchar4* d_pixels = nullptr; // A CUDA pointer mapped to the D3D12 texture's memory

    // D3D12/CUDA interop objects
    Microsoft::WRL::ComPtr<ID3D12Resource> d_texture;
    HANDLE sharedHandle = nullptr;
    cudaExternalMemory_t cudaExtMem = nullptr;
};
#else

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

#endif