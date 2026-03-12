#ifdef _WIN32
#include "renderer/D3D12/D3D12Device.hpp"
#include "State.hpp"
#include <stdexcept>
#include <iostream>
#if defined(HAS_STREAMLINE)
#include <sl.h>
#include <sl_dlss.h>
#endif
#include <d3d12/d3dx12.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#if defined(HAS_STREAMLINE)
#pragma comment(lib, "sl.interposer.lib")
#endif

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

// Helper for checking HRESULTs
static void checkHresult(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        std::cerr << "D3D12/DXGI Error (" << msg << ")" << std::endl;
        throw std::runtime_error(msg);
    }
}

// Helper for Streamline results
#if defined(HAS_STREAMLINE)
static inline void successCheck(sl::Result result, const char* msg) {
    if(result != sl::Result::eOk) {
        std::cerr << "sl error (" << msg << "): " << (int)result << "\n";
        throw std::runtime_error(msg);
    }
}
#endif


D3D12Device::D3D12Device()
 :  frameIndex(0), rtvDescriptorSize(0), fenceValue(1), fenceEvent(nullptr), cudaSyncSemaphore{}
{
    for(UINT i = 0; i < g_frameCount; ++i) {
        fenceValues[i] = 0;
    }
}

D3D12Device::~D3D12Device() {
    WaitForGpu();
    if (cudaSyncSemaphore) {
        cudaDestroyExternalSemaphore(cudaSyncSemaphore);
    }
    CloseHandle(fenceEvent);
#if defined(HAS_STREAMLINE)
    slShutdown();
#endif
}

void D3D12Device::Initialize(void* windowHandle) {
    HWND hwnd = static_cast<HWND>(windowHandle);

    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) {
        throw std::runtime_error("Failed to initialize COM library.");
    }
    AllocConsole();
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);

    // --- DXGI Factory & Debug Layer ---
    UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
    {
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            std::cout << "[INFO] D3D12 Debug Layer Enabled.\n";
        }
    }
#endif
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    checkHresult(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2 failed.");

    // --- Adapter & Device ---
    Microsoft::WRL::ComPtr<IDXGIAdapter1> hardwareAdapter;
    LUID adapterLUID = {};
    for (UINT i = 0; factory->EnumAdapters1(i, &hardwareAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        hardwareAdapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
            adapterLUID = desc.AdapterLuid;
            printf("Found suitable adapter: %S\n", desc.Description);
            break;
        }
    }
    if (!hardwareAdapter) throw std::runtime_error("Failed to find a suitable D3D12 adapter.");
    checkHresult(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3dDevice)), "D3D12CreateDevice failed.");

    // --- Match to CUDA Device ---
    int cudaDeviceCount = 0;
    cudaGetDeviceCount(&cudaDeviceCount);
    int cudaDevice = -1;
    for (int i = 0; i < cudaDeviceCount; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        if (memcmp(&adapterLUID, devProp.luid, sizeof(LUID)) == 0) {
            cudaDevice = i;
            printf("Found matching CUDA device #%d: %s\n", i, devProp.name);
            break;
        }
    }
    if (cudaDevice == -1) throw std::runtime_error("Could not find a matching CUDA device.");
    cudaSetDevice(cudaDevice);

    // --- Streamline ---
#if defined(HAS_STREAMLINE)
    sl::Preferences prefs = {};
    prefs.renderAPI = sl::RenderAPI::eD3D12;
    // Fill in other prefs...
    const sl::Feature features[] = { sl::kFeatureDLSS };
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = _countof(features);
    successCheck(slInit(prefs, sl::kSDKVersion), "slInit");
    successCheck(slSetD3DDevice(d3dDevice.Get()), "slSetD3DDevice");
#endif

    // --- Command Queue & Swap Chain ---
    D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    checkHresult(d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)), "CreateCommandQueue failed.");

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = g_frameCount;
    swapChainDesc.Width = State::screenWIDTH;
    swapChainDesc.Height = State::screenHEIGHT;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    checkHresult(factory->CreateSwapChainForHwnd(commandQueue.Get(), hwnd, &swapChainDesc, nullptr, nullptr, &swapChain1), "CreateSwapChainForHwnd failed.");
    swapChain1.As(&swapChain);

    // --- RTVs and Command Objects ---
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = g_frameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    checkHresult(d3dDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)), "Create RTV heap failed.");
    rtvDescriptorSize = d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < g_frameCount; n++) {
        swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n]));
        d3dDevice->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, rtvDescriptorSize);
    }
    frameIndex = swapChain->GetCurrentBackBufferIndex();
    
    for (UINT n = 0; n < g_frameCount; n++) {
        checkHresult(d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocators[n])), "CreateCommandAllocator failed.");
    }
    checkHresult(d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[0].Get(), nullptr, IID_PPV_ARGS(&commandList)), "CreateCommandList failed.");
    commandList->Close();

    // --- Fences & Interop Semaphore ---
    checkHresult(d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)), "CreateFence failed.");
    HANDLE fenceHandle = nullptr;
    checkHresult(d3dDevice->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &fenceHandle), "CreateSharedHandle for fence failed.");
    cudaExternalSemaphoreHandleDesc semHandleDesc = {};
    semHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semHandleDesc.handle.win32.handle = fenceHandle;
    cudaImportExternalSemaphore(&cudaSyncSemaphore, &semHandleDesc);
    CloseHandle(fenceHandle);
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}

void D3D12Device::BeginFrame() {
    commandAllocators[frameIndex]->Reset();
    commandList->Reset(commandAllocators[frameIndex].Get(), nullptr);
}

void D3D12Device::EndFrame() {
    checkHresult(commandList->Close(), "Failed to close command list");
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    swapChain->Present(1, 0);
    MoveToNextFrame();
}

ID3D12Resource* D3D12Device::GetCurrentBackBuffer() {
    return renderTargets[frameIndex].Get();
}

void D3D12Device::MoveToNextFrame() {
    const UINT64 currentFenceValue = fenceValue;
    commandQueue->Signal(fence.Get(), currentFenceValue);
    fenceValue++;

    frameIndex = swapChain->GetCurrentBackBufferIndex();

    if (fenceValues[frameIndex] != 0 && fence->GetCompletedValue() < fenceValues[frameIndex]) {
        fence->SetEventOnCompletion(fenceValues[frameIndex], fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    fenceValues[frameIndex] = currentFenceValue;
}

void D3D12Device::WaitForGpu() {
    if (fenceValue == 0) return;  // Guard against fence value 0
    commandQueue->Signal(fence.Get(), fenceValue);
    fence->SetEventOnCompletion(fenceValue, fenceEvent);
    WaitForSingleObject(fenceEvent, INFINITE);
    fenceValue++;
}

#endif // _WIN32
