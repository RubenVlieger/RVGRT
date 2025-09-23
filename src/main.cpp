#define CONSOLE

#ifdef D3D12
#include <initguid.h>

//#define FULLSCREEN
#define NOMINMAX  // Prevent windows.h from defining min/max macros


#include <windows.h>

#include <sl.h>
#include <sl_dlss.h>   // DLSS feature
#include <sl_consts.h> // enums, structures
#include <sl_security.h>
#include <sl_version.h>
#include <sl_core_types.h>
#include <sl_core_api.h>

#include "State.hpp"    
#include "StateRender.cuh"
#include <windowsx.h>
#include <chrono>
#include "Timer.hpp"
#include <thread>
#include <atomic>
#include <vector>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxgi1_4.h>
#include <d3d12/d3dx12.h>



#include <wrl.h>
#include <hidsdi.h>
#include "Texturepack.cuh"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib") 
#pragma comment(lib, "sl.interposer.lib")

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

sl::ViewportHandle m_viewport = {123};

std::atomic<bool> running = true;

using std::cout;
using std::cerr;
using std::endl;
using Microsoft::WRL::ComPtr;

const char g_szClassName[] = "myWindowClass";
const UINT g_frameCount = 2;

RECT windowRect;
sl::DLSSMode g_dlssMode = sl::DLSSMode::eBalanced;

static inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error(msg);
    }
}
static inline void successCheck(sl::Result result, const char* msg) {

    if(result != sl::Result::eOk)
    {
        std::cerr << "sl error (" << msg << "): " << (int)result << "\n";
        throw std::runtime_error(msg);
    }
}

void MoveToNextFrame()
{
    // Schedule a Signal command in the queue for the frame we just submitted.
    const UINT64 fenceValueForThisFrame = State::state.fenceValue;
    State::state.commandQueue->Signal(State::state.fence.Get(), fenceValueForThisFrame);

    // Store this fence value against the back buffer index we just used.
    // This is crucial for knowing when this frame's work is truly finished.
    State::state.fenceValues[State::state.frameIndex] = fenceValueForThisFrame;

    // Increment the master fence value for the next frame.
    State::state.fenceValue++;

    // Update the frame index to the next available back buffer.
    State::state.frameIndex = State::state.swapChain->GetCurrentBackBufferIndex();

    // Now, wait for the GPU to finish any previous work on this *new* back buffer.
    // The value we wait for is the one we stored the last time this buffer was used.
    if (State::state.fence->GetCompletedValue() < State::state.fenceValues[State::state.frameIndex])
    {
        State::state.fence->SetEventOnCompletion(State::state.fenceValues[State::state.frameIndex], State::state.fenceEvent);
        WaitForSingleObjectEx(State::state.fenceEvent, INFINITE, FALSE);
    }
}



void renderLoop()
{
    using clock = std::chrono::steady_clock;
    auto lastTime = clock::now();
    double frameTimeMs = 16.6f;
    unsigned int frameCount = 0;

    while (running)
    {
        sl::FrameToken* frameToken;
        successCheck(slGetNewFrameToken(frameToken , &frameCount), "slGetNewFrameToken");
        ID3D12CommandAllocator* currentCommandAllocator = State::state.commandAllocators[State::state.frameIndex].Get();
        currentCommandAllocator->Reset();
        State::state.commandList->Reset(currentCommandAllocator, nullptr);

        State::state.render->GIdata.UpdateGIData(State::state.render->cArray, State::state.render->csdf, State::state.render->texturepack);

        // --- Perform CUDA Rendering at Low Resolution ---
        State::state.character.Update(frameCount);
        State::state.deltaTime = (float)frameTimeMs / 1000.f;
        State::state.render->drawCUDA(
            State::state.character.camera.pos,
            State::state.character.camera.forward,
            State::state.character.camera.up,
            State::state.character.camera.right,
            &State::state.character.unjitteredViewProjectionMatrix,
            &State::state.character.prevUnjitteredViewProjectionMatrix,
            State::state.character.jitterX,
            State::state.character.jitterY);

        sl::Constants consts = {};

        consts.reset = frameCount == 0 ? sl::Boolean::eTrue : sl::Boolean::eFalse;

        consts.jitterOffset = {State::state.character.jitterX, State::state.character.jitterY};
        consts.mvecScale = {1.0f, 1.0f};//{ 0.5f * (float)State::state.dispWIDTH, 0.5f * (float)State::state.dispHEIGHT };

        consts.cameraPos = {State::state.character.camera.pos.x, State::state.character.camera.pos.y, State::state.character.camera.pos.z};
        consts.cameraUp = {State::state.character.camera.up.x, State::state.character.camera.up.y, State::state.character.camera.up.z};
        consts.cameraRight = {State::state.character.camera.right.x, State::state.character.camera.right.y, State::state.character.camera.right.z};
        consts.cameraFwd = {State::state.character.camera.forward.x, State::state.character.camera.forward.y, State::state.character.camera.forward.z};
        consts.cameraNear = State::state.character.nearPlane;
        consts.cameraFar = State::state.character.farPlane;
        consts.cameraFOV = glm::radians(State::state.character.FOV);
        consts.cameraAspectRatio = (float)State::state.dispWIDTH / (float)State::state.dispHEIGHT;

        consts.depthInverted = sl::Boolean::eFalse;
        consts.cameraMotionIncluded = sl::Boolean::eFalse; // We provide matrices, so SL will handle camera motion.
        consts.motionVectors3D = sl::Boolean::eFalse;
        consts.orthographicProjection = sl::Boolean::eFalse;
        
        glm::mat4 viewProj = State::state.character.unjitteredViewProjectionMatrix;
        glm::mat4 prevViewProj = State::state.character.prevUnjitteredViewProjectionMatrix; 
        glm::mat4 clipToPrevClip = prevViewProj * glm::inverse(viewProj);

        glm::mat4 cameraViewToClip_T = glm::transpose(State::state.character.projectionMatrix);
        glm::mat4 clipToCameraView_T = glm::transpose(glm::inverse(State::state.character.projectionMatrix));
        glm::mat4 clipToPrevClip_T   = glm::transpose(clipToPrevClip);
        glm::mat4 prevClipToClip_T   = glm::transpose(glm::inverse(clipToPrevClip));

        // Now we can safely take the address of these local variables for memcpy
        memcpy(&consts.cameraViewToClip, &cameraViewToClip_T, sizeof(glm::mat4));
        memcpy(&consts.clipToCameraView, &clipToCameraView_T, sizeof(glm::mat4));
        memcpy(&consts.clipToPrevClip,   &clipToPrevClip_T,   sizeof(glm::mat4));
        memcpy(&consts.prevClipToClip,   &prevClipToClip_T,   sizeof(glm::mat4));

        successCheck(slSetConstants(consts, *frameToken, m_viewport), "slSetConstants");
    
        ID3D12Resource* inputTexture   = State::state.render->lowResColorBuffer.GetD3D12Resource();
        ID3D12Resource* mvTexture      = State::state.render->motionVectorTex.GetD3D12Resource();
        ID3D12Resource* depthTexture   = State::state.render->depthTex.GetD3D12Resource();
        ID3D12Resource* outputTexture  = State::state.render->upscaledColorBuffer.GetD3D12Resource();
        ID3D12Resource* backBuffer     = State::state.renderTargets[State::state.frameIndex].Get();

        D3D12_RESOURCE_BARRIER barriers[] = {
            // Inputs must be readable by the shader
            CD3DX12_RESOURCE_BARRIER::Transition(inputTexture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(mvTexture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(depthTexture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            // Output will be written to by a compute shader (UAV)
            CD3DX12_RESOURCE_BARRIER::Transition(outputTexture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        State::state.commandList->ResourceBarrier(_countof(barriers), barriers);

            sl::ViewportHandle view(m_viewport);
        const sl::BaseStructure* baseinputs[] = { &view };

        successCheck(slEvaluateFeature(sl::kFeatureDLSS, *frameToken, baseinputs, _countof(baseinputs), State::state.commandList.Get()), "slEvaluateFeature_DLSS");
        // --- 6. Displaying the Final Image on Screen ---
        // First, transition the DLSS output to be a copy source, and the back buffer to be a copy destination.
        D3D12_RESOURCE_BARRIER copyBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(outputTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(backBuffer,    D3D12_RESOURCE_STATE_PRESENT,        D3D12_RESOURCE_STATE_COPY_DEST)
        };
        State::state.commandList->ResourceBarrier(_countof(copyBarriers), copyBarriers);

        // Execute the copy from the DLSS output texture into the swap chain's back buffer.
        State::state.commandList->CopyResource(backBuffer, outputTexture);

        // Finally, transition the back buffer to the PRESENT state, so it can be shown on the monitor.
        // Also transition the output texture back to its default state for the next frame.
        D3D12_RESOURCE_BARRIER finalBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(outputTexture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            CD3DX12_RESOURCE_BARRIER::Transition(backBuffer,    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)
        };
        State::state.commandList->ResourceBarrier(_countof(finalBarriers), finalBarriers);

        // --- Finish and Execute ---
        State::state.commandList->Close();
        ID3D12CommandList* ppCommandLists[] = { State::state.commandList.Get() };
        State::state.commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

        // --- Present and Sync ---
        State::state.swapChain->Present(1, 0);

        MoveToNextFrame();

        // --- Frame Timing ---
        frameTimeMs = std::chrono::duration<double, std::milli>(clock::now() - lastTime).count();
        State::state.frameTimeAverager.addFrameTime(frameTimeMs);
        lastTime = clock::now();

        char title[200];
        snprintf(title, sizeof(title),
                 "FSR 1.0 Demo: %.1f ms - avg: %.1f ms (Render: %dx%d -> Display: %dx%d)",
                 frameTimeMs, State::state.frameTimeAverager.getAverage(),
                 State::screenWIDTH, State::screenHEIGHT, State::dispWIDTH, State::dispHEIGHT);
        SetWindowTextA(State::state.hwnd, title);
        frameCount++;
    }
}




void WndCreate(HWND hwnd)
{
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) {
        throw std::runtime_error("Failed to initialize COM library.");
    }


    // --- 0. PRE-INITIALIZATION ---
#ifdef CONSOLE
    AllocConsole();
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    printf("Console Initialized.\n");
#endif



    // --- 0.5 

    // --- 1. DXGI FACTORY AND DEBUG LAYER ---
    printf("Initializing D3D12 and DXGI...\n");
    UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();

            // Set the flag to create the factory with debug enabled.
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            printf("[INFO] D3D12 Debug Layer Enabled.\n");
        }
        else
        {
            printf("[WARN] Failed to get the D3D12 debug interface. Is 'Graphics Tools' installed?\n");
        }
    }
#endif


    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)))) {
        throw std::runtime_error("Failed to create DXGI Factory.");
    }

    // --- 2. ADAPTER AND DEVICE SELECTION ---
    printf("Selecting physical adapter...\n");
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    LUID adapterLUID = {}; // The unique ID of the chosen GPU
    for (UINT i = 0; factory->EnumAdapters1(i, &hardwareAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        hardwareAdapter->GetDesc1(&desc);

        sl::AdapterInfo adapterInfo{};
        adapterInfo.deviceLUID = (uint8_t*)&desc.AdapterLuid;
        adapterInfo.deviceLUIDSizeInBytes = sizeof(LUID);


//        std::cout << "Is DLSS SUPPORTED: " << SUCCEEDED(slIsFeatureSupported(sl::kFeatureDLSS, adapterInfo)) << std::endl;

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
            adapterLUID = desc.AdapterLuid;
            printf("Found suitable adapter: %S\n", desc.Description);
            break;
        }
    }
    if (!hardwareAdapter) throw std::runtime_error("Failed to find a suitable D3D12 adapter.");

    if (FAILED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&State::state.d3dDevice)))) {
        throw std::runtime_error("Failed to create D3D12 device.");
    }

    // --- 3. CUDA DEVICE MATCHING ---
    printf("Matching D3D12 adapter to CUDA device...\n");
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
    if (cudaDevice == -1) throw std::runtime_error("Could not find a matching CUDA device for the D3D12 adapter.");
    cudaSetDevice(cudaDevice);

        D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    if (SUCCEEDED(State::state.d3dDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
    {
        std::cout << "[INFO] GPU supports TiledResourcesTier TIER_" << options.TiledResourcesTier - 1<< std::endl;
        if (options.StandardSwizzle64KBSupported) {
            printf("[INFO] GPU supports StandardSwizzle64KB layout.\n");
        } else {
            printf("[WARN] GPU does NOT support StandardSwizzle64KB layout. A copy is required for interop.\n");
        }
    }


    
    printf("Initializing Streamline DLSS...\n");

    sl::Preferences prefs = {};
    prefs.renderAPI = sl::RenderAPI::eD3D12;
    prefs.showConsole = true;
    prefs.logLevel = sl::LogLevel::eVerbose;

    const sl::Feature features[] = { sl::kFeatureDLSS };
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = _countof(features);
    prefs.engineVersion = "1.1";
    prefs.engine = sl::EngineType::eCustom;
    prefs.projectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";

    successCheck(slInit(prefs, sl::kSDKVersion), "slInit");

    successCheck(slSetD3DDevice(State::state.d3dDevice.Get()), "slSetD3DDevice");


    // --- 4. COMMAND QUEUE, SWAP CHAIN, AND RENDER TARGETS ---
    printf("Creating command queue..\n");
    D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    if (FAILED(State::state.d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&State::state.commandQueue)))) {
        throw std::runtime_error("Failed to create command queue.");
    }

    /// swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = g_frameCount;
    swapChainDesc.Width = State::screenWIDTH;
    swapChainDesc.Height = State::screenHEIGHT;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swapChain1;
    if (FAILED(factory->CreateSwapChainForHwnd(State::state.commandQueue.Get(), hwnd, &swapChainDesc, nullptr, nullptr, &swapChain1))) {
        throw std::runtime_error("Failed to create swap chain.");
    }
    swapChain1.As(&State::state.swapChain);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    State::state.frameIndex = State::state.swapChain->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = g_frameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    State::state.d3dDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&State::state.rtvHeap));
    State::state.rtvDescriptorSize = State::state.d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(State::state.rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < g_frameCount; n++) {
        State::state.swapChain->GetBuffer(n, IID_PPV_ARGS(&State::state.renderTargets[n]));
        State::state.d3dDevice->CreateRenderTargetView(State::state.renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, State::state.rtvDescriptorSize);
    }

    // --- 5. COMMAND ALLOCATORS AND LIST ---
    printf("Creating command allocators and list...\n");
    for (UINT n = 0; n < g_frameCount; n++) {
        State::state.d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&State::state.commandAllocators[n]));
    }
    State::state.d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, State::state.commandAllocators[0].Get(), nullptr, IID_PPV_ARGS(&State::state.commandList));
    State::state.commandList->Close();

    // --- 6. SYNCHRONIZATION: THE SHARED D3D12-CUDA FENCE ---
    printf("Creating shared D3D12-CUDA synchronization fence...\n");
    if (FAILED(State::state.d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&State::state.fence)))) {
        throw std::runtime_error("Failed to create shared D3D12 fence.");
    }
    
    HANDLE fenceHandle = nullptr;
    if (FAILED(State::state.d3dDevice->CreateSharedHandle(State::state.fence.Get(), nullptr, GENERIC_ALL, nullptr, &fenceHandle))) {
        throw std::runtime_error("Failed to create shared handle for fence.");
    }
    
    cudaExternalSemaphoreHandleDesc semHandleDesc = {};
    semHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semHandleDesc.handle.win32.handle = fenceHandle;
    cudaImportExternalSemaphore(&State::state.cudaSyncSemaphore, &semHandleDesc);
    CloseHandle(fenceHandle); // We can close the handle now that CUDA has imported it.

    State::state.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    for (UINT i = 0; i < g_frameCount; ++i) { State::state.fenceValues[i] = 0; }
    State::state.fenceValue = 1;


    // --- 7. INITIALIZE ALL SHARED D3D12-CUDA RESOURCES ---
    printf("Initializing shared render textures...\n");
    State::state.render->lowResColorBuffer.Initialize(State::state.d3dDevice.Get(), State::dispWIDTH, State::dispHEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER, L"LowResColorBuffer");
    State::state.render->upscaledColorBuffer.Initialize_D3D12_Only(State::state.d3dDevice.Get(), 
                                                                    State::screenWIDTH, 
                                                                    State::screenHEIGHT, 
                                                                    DXGI_FORMAT_R8G8B8A8_UNORM, 
                                                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, // The required flag for DLSS output
                                                                    L"UpscaledColorBuffer"
                                                                );
    State::state.render->motionVectorTex.Initialize(State::state.d3dDevice.Get(), State::dispWIDTH, State::dispHEIGHT, DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER, L"MotionVectorTexture");
    State::state.render->depthTex.Initialize(State::state.d3dDevice.Get(), State::dispWIDTH, State::dispHEIGHT, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,L"DepthTexture");
    State::state.render->shadowTex.Initialize_Cuda_Array(State::state.dispWIDTH / 2, State::state.dispHEIGHT / 2, cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat), cudaFilterModeLinear);
    State::state.render->halfDistBuffer.Initialize_Cuda_Array(State::state.dispWIDTH / 2, State::state.dispHEIGHT / 2, cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat), cudaFilterModePoint);

    // State::state.render->shadowTex.Initialize(State::state.d3dDevice.Get(), State::dispWIDTH / 2, State::dispHEIGHT / 2, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,L"ShadowTexture");
    // State::state.render->halfDistBuffer.Initialize(State::state.d3dDevice.Get(), State::dispWIDTH / 2, State::dispHEIGHT / 2, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,L"HalfDistTexture");

     printf("Tagging resources for Streamline...\n");

    // CRITICAL: The command list must be open to record commands for tagging.
    // We can use the first frame's command allocator.
    State::state.commandAllocators[0]->Reset();
    State::state.commandList->Reset(State::state.commandAllocators[0].Get(), nullptr);

    ID3D12Resource* colorInResource  = State::state.render->lowResColorBuffer.GetD3D12Resource();
    ID3D12Resource* colorOutResource = State::state.render->upscaledColorBuffer.GetD3D12Resource();
    ID3D12Resource* depthInResource  = State::state.render->depthTex.GetD3D12Resource();
    ID3D12Resource* mvInResource     = State::state.render->motionVectorTex.GetD3D12Resource();

    // Step 2: Get the descriptions for each resource to get width, height, format
    D3D12_RESOURCE_DESC colorInDesc  = colorInResource->GetDesc();
    D3D12_RESOURCE_DESC colorOutDesc = colorOutResource->GetDesc();
    D3D12_RESOURCE_DESC depthInDesc  = depthInResource->GetDesc();
    D3D12_RESOURCE_DESC mvInDesc     = mvInResource->GetDesc();

    // Step 3: FULLY initialize the sl::Resource structs
    sl::Resource colorIn  = {}; // Use {} to zero-initialize all members first
    colorIn.type = sl::ResourceType::eTex2d;
    colorIn.native = colorInResource;
    colorIn.state = D3D12_RESOURCE_STATE_COMMON; // State at the moment of this call
    colorIn.width = colorInDesc.Width;
    colorIn.height = colorInDesc.Height;
    colorIn.nativeFormat = colorInDesc.Format;

    sl::Resource colorOut = {};
    colorOut.type = sl::ResourceType::eTex2d;
    colorOut.native = colorOutResource;
    colorOut.state = D3D12_RESOURCE_STATE_COMMON;
    colorOut.width = colorOutDesc.Width;
    colorOut.height = colorOutDesc.Height;
    colorOut.nativeFormat = colorOutDesc.Format;

    sl::Resource depthIn = {};
    depthIn.type = sl::ResourceType::eTex2d;
    depthIn.native = depthInResource;
    depthIn.state = D3D12_RESOURCE_STATE_COMMON;
    depthIn.width = depthInDesc.Width;
    depthIn.height = depthInDesc.Height;
    depthIn.nativeFormat = depthInDesc.Format;

    sl::Resource mvIn = {};
    mvIn.type = sl::ResourceType::eTex2d;
    mvIn.native = mvInResource;
    mvIn.state = D3D12_RESOURCE_STATE_COMMON;
    mvIn.width = mvInDesc.Width;
    mvIn.height = mvInDesc.Height;
    mvIn.nativeFormat = mvInDesc.Format;

    // Step 4: Create the ResourceTags pointing to our FULLY populated structs
    sl::ResourceTag colorInTag  = {&colorIn,  sl::kBufferTypeScalingInputColor,  sl::ResourceLifecycle::eValidUntilPresent};
    sl::ResourceTag colorOutTag = {&colorOut, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilPresent};
    sl::ResourceTag depthTag    = {&depthIn,  sl::kBufferTypeDepth,              sl::ResourceLifecycle::eValidUntilPresent};
    sl::ResourceTag mvecTag     = {&mvIn,     sl::kBufferTypeMotionVectors,      sl::ResourceLifecycle::eValidUntilPresent};

    // Step 5: Put tags in an array and call slSetTag once
    sl::ResourceTag tags[] = {colorInTag, colorOutTag, depthTag, mvecTag};
    successCheck(slSetTag(m_viewport, tags, _countof(tags), State::state.commandList.Get()), "slSetTag Global");
    // You can now close the command list so it's ready for the first frame.
    State::state.commandList->Close();
    // sl::Resource colorIn  = {sl::ResourceType::eTex2d, State::state.render->lowResColorBuffer.GetD3D12Resource(),  sl::ResourceLifecycle::eValidUntilPresent};
    // sl::Resource mvIn     = {sl::ResourceType::eTex2d, State::state.render->motionVectorTex.GetD3D12Resource(),     sl::ResourceLifecycle::eValidUntilPresent};
    // sl::Resource depthIn  = {sl::ResourceType::eTex2d, State::state.render->depthTex.GetD3D12Resource(),  sl::ResourceLifecycle::eValidUntilPresent };
    // sl::Resource colorOut = {sl::ResourceType::eTex2d, State::state.render->upscaledColorBuffer.GetD3D12Resource(), sl::ResourceLifecycle::eValidUntilPresent };
    

    // sl::ResourceTag colorInTag = sl::ResourceTag {&colorIn, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eOnlyValidNow };
    // sl::ResourceTag colorOutTag = sl::ResourceTag {&colorOut, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eOnlyValidNow };
    // sl::ResourceTag depthTag = sl::ResourceTag {&depthIn, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent };
    // sl::ResourceTag mvecTag = sl::ResourceTag {&mvIn, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow };

    // Tag them globally ONCE
    // The viewport handle m_viewport is the same one you use in the render loop
    // successCheck(slSetTag(m_viewport, &colorInTag,  sl::kBufferTypeScalingInputColor, State::state.commandList.Get()), "slSetTag colorIn");
    // successCheck(slSetTag(m_viewport, &colorOutTag, sl::kBufferTypeScalingOutputColor, State::state.commandList.Get()), "slSetTag colorOut");
    // successCheck(slSetTag(m_viewport, &depthTag,  sl::kBufferTypeDepth, State::state.commandList.Get()), "slSetTag depthIn");
    // successCheck(slSetTag(m_viewport, &mvecTag,     sl::kBufferTypeMotionVectors, State::state.commandList.Get()), "slSetTag mvIn");


    // Check if DLSS is available
    sl::FeatureRequirements dlssReq;
    successCheck(slGetFeatureRequirements(sl::kFeatureDLSS, dlssReq), "slGetFeatureRequirements");
    // Setup DLSS
    sl::DLSSOptimalSettings dlssSettings;
    
    sl::DLSSOptions dlssOptions;
    // These are populated based on user selection in the UI
    dlssOptions.mode = sl::DLSSMode::eUltraPerformance;
    dlssOptions.outputWidth = State::state.screenWIDTH;    
    dlssOptions.outputHeight = State::state.screenHEIGHT;
    // Now let's check what should our rendering resolution be

        // Handle error here
    successCheck(slDLSSSetOptions(m_viewport, dlssOptions), "slDLSSSetOptions"); 

    printf("Streamline DLSS initialized successfully.\n");

    // --- 9. Continue app init ---
    State::state.Create();
    printf("Initialization Complete.\n");

}



LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) 
{
    switch(msg) {
        case WM_KEYDOWN: {
            unsigned long KC = (unsigned long)wParam;
            State::state.keysPressed.set(KC, 1);

           if (KC == VK_ESCAPE) {
                CloseWindow(hwnd);
                exit(0);
            }
            break;
        }
        case WM_KEYUP: {
            unsigned char KC = (unsigned char)wParam;
            State::state.keysPressed.set(KC, 0);
            break;
        }
        case WM_CREATE:
            WndCreate(hwnd);
            break;
        case WM_SIZE:
            GetWindowRect(hwnd, &windowRect);
            break;
        case WM_INPUT: {
            // Read RAWINPUT deltas only
            UINT size;
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER));
            static std::vector<BYTE> rawBuffer(size);
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, rawBuffer.data(), &size, sizeof(RAWINPUTHEADER));
            RAWINPUT* raw = (RAWINPUT*)rawBuffer.data();
            if (raw->header.dwType == RIM_TYPEMOUSE) {
                long mouseX = raw->data.mouse.lLastX;
                long mouseY = raw->data.mouse.lLastY;
                State::state.deltaXMouse.fetch_add(mouseX, std::memory_order_relaxed);
                State::state.deltaYMouse.fetch_add(mouseY, std::memory_order_relaxed);
            }
            break;
        }
        case WM_CLOSE:
            DestroyWindow(hwnd);

            break;
        case WM_DESTROY:
            MoveToNextFrame();
            CloseHandle(State::state.fenceEvent);
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) 
{
    WNDCLASSEX wc = {};
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszClassName = g_szClassName;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&wc)) {
        MessageBox(NULL, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

#ifdef FULLSCREEN
    int screenWidth  = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    State::state.hwnd = CreateWindowEx(
        0, g_szClassName, "Ruben leip programma",
        WS_POPUP, 0, 0, screenWidth, screenHeight,
        NULL, NULL, hInstance, NULL
    );
#else
    State::state.hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE, g_szClassName, "Ruben leip programma",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, State::screenWIDTH, State::screenHEIGHT,
        NULL, NULL, hInstance, NULL
    );
#endif

    if (!State::state.hwnd) {
        MessageBox(NULL, "Window Creation Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // Register RAWINPUT
    RAWINPUTDEVICE rid[1] = {};
    rid[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
    rid[0].usUsage     = HID_USAGE_GENERIC_MOUSE;
    rid[0].dwFlags     = RIDEV_INPUTSINK | RIDEV_NOLEGACY;
    rid[0].hwndTarget  = State::state.hwnd;
    RegisterRawInputDevices(rid, 1, sizeof(rid[0]));

    ShowWindow(State::state.hwnd, nCmdShow);
    UpdateWindow(State::state.hwnd);
    GetWindowRect(State::state.hwnd, &windowRect);

    std::thread renderThread(renderLoop);

    MSG Msg;
    while (GetMessage(&Msg, NULL, 0, 0) > 0) {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    running = false;
    renderThread.join();

    return (int)Msg.wParam;
}
#else

#define CONSOLE
//#define FULLSCREEN
#define NOMINMAX  // Prevent windows.h from defining min/max macros
#include <windows.h>
#include "State.hpp"    
#include "StateRender.cuh"
#include <windowsx.h>
#include <chrono>
#include "Timer.hpp"
#include <thread>
#include <atomic>
#include <vector>
#include "d3d11.h"
#include <hidsdi.h>
#include "Texturepack.cuh"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib") 

std::atomic<bool> running = true;

using std::cout;
using std::cerr;
using std::endl;
const char g_szClassName[] = "myWindowClass";

RECT windowRect;

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

void renderLoop() 
{
    using clock = std::chrono::steady_clock;

    auto lastTime = clock::now();

    while (running) 
    {
        State::state.render->GIdata.UpdateGIData(State::state.render->cArray, State::state.render->csdf, State::state.render->texturepack);

        // Update game state
        State::state.character.Update();
        State::state.deltaTime = 0.016f;

        // Render CUDA framebuffer
        State::state.render->drawCUDA(
            State::state.character.camera.pos,
            State::state.character.camera.forward,
            State::state.character.camera.up,
            State::state.character.camera.right
        );

        // Copy CUDA D3D texture to backbuffer
        State::state.render->framebuffer.CopyDeviceToTexture();
        ID3D11DeviceContext* ctx = State::state.d3dContext;
        ctx->CopyResource(State::state.backBufferTexture, State::state.render->framebuffer.getD3DTexture());

        // Present
        State::state.swapChain->Present(1, 0);

        // Frame timing
        double frameTimeMs = std::chrono::duration<double, std::milli>(clock::now() - lastTime).count();
        State::state.frameTimeAverager.addFrameTime(frameTimeMs);
        lastTime = clock::now();

        // Update window title
        char title[200];
        snprintf(title, sizeof(title),
                 "Ruben leip programma: %.1f ms - average: %.1f ms",
                 frameTimeMs, State::state.frameTimeAverager.getAverage());
        SetWindowTextA(State::state.hwnd, title);
    }
}

void WndCreate(HWND hwnd) 
{
#ifdef CONSOLE
    AllocConsole();
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
#endif

    // Step 1: Create D3D11 device and swap chain
    DXGI_SWAP_CHAIN_DESC scd = {};
    scd.BufferCount = 1;
    scd.BufferDesc.Width  = State::dispWIDTH;
    scd.BufferDesc.Height = State::dispHEIGHT;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferDesc.RefreshRate.Numerator = 0; // auto
    scd.BufferDesc.RefreshRate.Denominator = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd;
    scd.SampleDesc.Count = 1;
#ifdef FULLSCREEN
    scd.Windowed = FALSE;
    scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
#else
    scd.Windowed = TRUE;
    scd.SwapEffect = DXGI_SWAP_EFFECT_SEQUENTIAL;
#endif

    D3D_FEATURE_LEVEL featureLevel;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &scd,
        &swapChain,
        &device,
        &featureLevel,
        &context
    );

    if (FAILED(hr)) {
        fprintf(stderr, "Failed to create D3D11 device and swap chain!\n");
        return;
    }

    // Backbuffer and RTV
    ID3D11Texture2D* backBuffer = nullptr;
    swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    ID3D11RenderTargetView* rtv = nullptr;
    device->CreateRenderTargetView(backBuffer, nullptr, &rtv);
    backBuffer->Release();

    // Store in State
    State::state.d3dDevice = device;
    State::state.d3dContext = context;
    State::state.swapChain = swapChain;
    State::state.backBufferTexture = backBuffer;

    // CUDA-D3D interop framebuffer
    State::state.render->framebuffer.Allocate(State::dispWIDTH, State::dispHEIGHT);
    State::state.render->framebuffer.InitializeInterop(device, context);

    // State init
    State::state.Create();

#//ifdef FULLSCREEN
    ShowCursor(FALSE); // hide cursor
//#endif
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) 
{
    switch(msg) {
        case WM_KEYDOWN: {
            unsigned long KC = (unsigned long)wParam;
            State::state.keysPressed.set(KC, 1);
            if(KC == VK_ESCAPE) DestroyWindow(hwnd);
            break;
        }
        case WM_KEYUP: {
            unsigned char KC = (unsigned char)wParam;
            State::state.keysPressed.set(KC, 0);
            break;
        }
        case WM_CREATE:
            WndCreate(hwnd);
            break;
        case WM_SIZE:
            GetWindowRect(hwnd, &windowRect);
            break;
        case WM_INPUT: {
            // Read RAWINPUT deltas only
            UINT size;
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER));
            static std::vector<BYTE> rawBuffer(size);
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, rawBuffer.data(), &size, sizeof(RAWINPUTHEADER));
            RAWINPUT* raw = (RAWINPUT*)rawBuffer.data();
            if (raw->header.dwType == RIM_TYPEMOUSE) {
                long mouseX = raw->data.mouse.lLastX;
                long mouseY = raw->data.mouse.lLastY;
                State::state.deltaXMouse.fetch_add(mouseX, std::memory_order_relaxed);
                State::state.deltaYMouse.fetch_add(mouseY, std::memory_order_relaxed);
            }
            break;
        }
        case WM_CLOSE:
            DestroyWindow(hwnd);
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) 
{
    WNDCLASSEX wc = {};
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszClassName = g_szClassName;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&wc)) {
        MessageBox(NULL, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

#ifdef FULLSCREEN
    int screenWidth  = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    State::state.hwnd = CreateWindowEx(
        0, g_szClassName, "Ruben leip programma",
        WS_POPUP, 0, 0, screenWidth, screenHeight,
        NULL, NULL, hInstance, NULL
    );
#else
    State::state.hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE, g_szClassName, "Ruben leip programma",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, State::dispWIDTH, State::dispHEIGHT,
        NULL, NULL, hInstance, NULL
    );
#endif

    if (!State::state.hwnd) {
        MessageBox(NULL, "Window Creation Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // Register RAWINPUT
    RAWINPUTDEVICE rid[1] = {};
    rid[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
    rid[0].usUsage     = HID_USAGE_GENERIC_MOUSE;
    rid[0].dwFlags     = RIDEV_INPUTSINK | RIDEV_NOLEGACY;
    rid[0].hwndTarget  = State::state.hwnd;
    RegisterRawInputDevices(rid, 1, sizeof(rid[0]));

    ShowWindow(State::state.hwnd, nCmdShow);
    UpdateWindow(State::state.hwnd);
    GetWindowRect(State::state.hwnd, &windowRect);

    std::thread renderThread(renderLoop);

    MSG Msg;
    while (GetMessage(&Msg, NULL, 0, 0) > 0) {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    running = false;
    renderThread.join();

    return (int)Msg.wParam;
}
#endif