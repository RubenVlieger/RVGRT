#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include "d3dx12/d3dx12.h"

class FSR1
{
public:
    FSR1();
    ~FSR1();

    // Call this during initialization
    void Initialize(
        ID3D12Device* device,
        UINT renderWidth, UINT renderHeight,
        UINT displayWidth, UINT displayHeight
    );

    // Call this every frame inside the render loop
    void Dispatch(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* inputTexture // The low-resolution texture from CUDA
    );
    
    // Getter for the final, upscaled texture
    ID3D12Resource* GetOutputTexture() { return m_outputTexture.Get(); }

private:
    void CreateResources(ID3D12Device* device);
    void CreatePipeline(ID3D12Device* device);

    // Resolutions
    UINT m_renderWidth;
    UINT m_renderHeight;
    UINT m_displayWidth;
    UINT m_displayHeight;

    // D3D12 Pipeline Objects
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
    Microsoft::WRL::ComPtr<ID3DBlob> m_shaderBlob;

    // FSR-specific D3D12 Resources
    Microsoft::WRL::ComPtr<ID3D12Resource> m_outputTexture; // The upscaled result
    Microsoft::WRL::ComPtr<ID3D12Resource> m_constBuffer;
    UINT8* m_constBufferData;

    // Descriptor Heaps for textures
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    UINT m_descriptorSize;

    // Handles to the descriptors
    CD3DX12_CPU_DESCRIPTOR_HANDLE m_inputSrvCpuHandle;
    CD3DX12_GPU_DESCRIPTOR_HANDLE m_inputSrvGpuHandle;
    CD3DX12_CPU_DESCRIPTOR_HANDLE m_outputUavCpuHandle;
    CD3DX12_GPU_DESCRIPTOR_HANDLE m_outputUavGpuHandle;
};