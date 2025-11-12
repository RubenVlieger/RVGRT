#pragma once

#ifdef _WIN32
#include "renderer/GraphicsDevice.hpp"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <cuda_runtime.h>

class D3D12Device : public GraphicsDevice
{
public:
    D3D12Device();
    ~D3D12Device() override;

    void Initialize(void* windowHandle) override;
    void BeginFrame() override;
    void EndFrame() override;

    ID3D12Device* GetD3D12Device() override { return d3dDevice.Get(); }
    ID3D12CommandQueue* GetCommandQueue() override { return commandQueue.Get(); }
    ID3D12Resource* GetCurrentBackBuffer();
    ID3D12GraphicsCommandList* GetCommandList() { return commandList.Get(); }
    UINT GetFrameIndex() { return frameIndex; }


private:
    void WaitForGpu();
    void MoveToNextFrame();

    static constexpr UINT g_frameCount = 2;

    Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
    Microsoft::WRL::ComPtr<ID3D12Resource> renderTargets[g_frameCount];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[g_frameCount];
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap;

    // Synchronization objects
    UINT frameIndex;
    UINT rtvDescriptorSize;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue;
    HANDLE fenceEvent;
    UINT64 fenceValues[g_frameCount];

public:
    // Make the interop semaphore public for easy access from the render loop
    cudaExternalSemaphore_t cudaSyncSemaphore;
};

#endif // _WIN32