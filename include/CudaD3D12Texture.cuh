#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <stdexcept>

class CudaD3D12Texture {
public:
    CudaD3D12Texture();
    ~CudaD3D12Texture();

    // Delete copy operations
    CudaD3D12Texture(const CudaD3D12Texture&) = delete;
    CudaD3D12Texture& operator=(const CudaD3D12Texture&) = delete;

    // Enable move semantics
    CudaD3D12Texture(CudaD3D12Texture&& other) noexcept;
    CudaD3D12Texture& operator=(CudaD3D12Texture&& other) noexcept;

    void Initialize(ID3D12Device* device,
                    UINT width,
                    UINT height,
                    DXGI_FORMAT format,
                    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    const wchar_t* debugName = L"SharedCudaD3D12Texture");

    void Initialize_D3D12_Only(ID3D12Device* device, 
                                UINT _width, 
                                UINT _height, 
                                DXGI_FORMAT format, 
                                D3D12_RESOURCE_FLAGS flags, 
                                const wchar_t* debugName);

    void Initialize_Cuda_Array(UINT _width, UINT _height, const cudaChannelFormatDesc& formatDesc, cudaTextureFilterMode filterMode);

    // D3D12 Getters
    ID3D12Resource* GetD3D12Resource() const { return m_d3dResource.Get(); }
    D3D12_RESOURCE_DESC GetDesc() const { return m_d3dResource ? m_d3dResource->GetDesc() : D3D12_RESOURCE_DESC{}; }

    // CUDA Getters
    void* GetCudaDevicePtr() const { return m_cudaDevPtr; }
    size_t getPitch() const { return m_pitch; }
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }

    cudaArray_t GetCudaArray() const { return m_cudaArray; }

    cudaSurfaceObject_t getCudaSurfObject() const { return m_cudaSurfaceObj; }
    cudaTextureObject_t getCudaTexObject() const { return m_cudaTextureObj; }

    bool IsValid() const { return m_d3dResource != nullptr && m_cudaExtMem != nullptr; }

private:
    size_t width;
    size_t height;
    void Release();

    // D3D12 Resources
    Microsoft::WRL::ComPtr<ID3D12Resource> m_d3dResource;
    Microsoft::WRL::ComPtr<ID3D12Heap> m_d3dHeap;

    cudaSurfaceObject_t m_cudaSurfaceObj = 0;
    cudaTextureObject_t m_cudaTextureObj = 0;

    // CUDA Interop Resources
    cudaExternalMemory_t m_cudaExtMem = nullptr;
    HANDLE m_sharedHandle = nullptr;
    
    // CUDA-side representations
    void* m_cudaDevPtr = nullptr;
    cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
    cudaArray_t m_cudaArray = nullptr;
    
    // Texture properties
    size_t m_pitch = 0;
};