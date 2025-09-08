#define CONSOLE

#ifdef D3D12
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
#include <d3d12.h>
#include "d3dx12/d3dx12.h"

#include <dxgi1_6.h>
#include <cuda_gl_interop.h>

#include <wrl.h>
#include <hidsdi.h>
#include "Texturepack.cuh"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib") 

std::atomic<bool> running = true;

using std::cout;
using std::cerr;
using std::endl;
using Microsoft::WRL::ComPtr;

const char g_szClassName[] = "myWindowClass";
const UINT g_frameCount = 2;

RECT windowRect;

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
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
// In main.cpp, replace the entire renderLoop with this corrected version

// This is the complete, corrected renderLoop function.

void renderLoop() 
{
    using clock = std::chrono::steady_clock;
    auto lastTime = clock::now();
    double frameTimeMs = 16.6f;
    while (running) 
    {
        // Get the command allocator for the current frame.
        ID3D12CommandAllocator* currentCommandAllocator = State::state.commandAllocators[State::state.frameIndex].Get();
        
        // Reset the allocator and the command list.
        currentCommandAllocator->Reset();
        State::state.commandList->Reset(currentCommandAllocator, nullptr);

        // --- Perform CUDA Rendering ---
        // This section remains unchanged.
        //State::state.render->GIdata.UpdateGIData(State::state.render->cArray, State::state.render->csdf, State::state.render->texturepack);
        State::state.character.Update();
        State::state.deltaTime = frameTimeMs / 1000;
        State::state.render->drawCUDA(
            State::state.character.camera.pos,
            State::state.character.camera.forward,
            State::state.character.camera.up,
            State::state.character.camera.right
        );

        ID3D12Resource* cudaTexture = State::state.render->framebuffer.getD3DTexture();

        // --- Record D3D12 Commands ---
        
        // 1. Transition the CUDA texture to be a copy source
        D3D12_RESOURCE_BARRIER barrier_common_to_copy_src = {};
        barrier_common_to_copy_src.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_common_to_copy_src.Transition.pResource = cudaTexture;
        barrier_common_to_copy_src.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier_common_to_copy_src.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier_common_to_copy_src.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        State::state.commandList->ResourceBarrier(1, &barrier_common_to_copy_src);

        // 2. Transition the back buffer to be a copy destination
        D3D12_RESOURCE_BARRIER barrier_pres_to_copy_dest = {};
        barrier_pres_to_copy_dest.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_pres_to_copy_dest.Transition.pResource = State::state.renderTargets[State::state.frameIndex].Get();
        barrier_pres_to_copy_dest.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier_pres_to_copy_dest.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier_pres_to_copy_dest.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        State::state.commandList->ResourceBarrier(1, &barrier_pres_to_copy_dest);

        // 3. Perform the robust texture-to-texture copy
        // THIS IS THE CORRECTED SECTION
        {
            // Describe the destination (the back buffer)
            D3D12_TEXTURE_COPY_LOCATION dest = {};
            dest.pResource = State::state.renderTargets[State::state.frameIndex].Get();
            dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dest.SubresourceIndex = 0;

            // Describe the source (your row-major CUDA texture)
            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = cudaTexture;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.SubresourceIndex = 0;

            // Perform the copy. The driver will correctly handle the conversion
            // from the row-major layout to the back buffer's optimized layout.
            State::state.commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
        }

        // 4. Transition the back buffer back to the present state
        D3D12_RESOURCE_BARRIER barrier_copy_dest_to_pres = {};
        barrier_copy_dest_to_pres.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_copy_dest_to_pres.Transition.pResource = State::state.renderTargets[State::state.frameIndex].Get();
        barrier_copy_dest_to_pres.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier_copy_dest_to_pres.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        barrier_copy_dest_to_pres.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        State::state.commandList->ResourceBarrier(1, &barrier_copy_dest_to_pres);
        
        // 5. Transition the CUDA texture back to its common state
        D3D12_RESOURCE_BARRIER barrier_copy_src_to_common = {};
        barrier_copy_src_to_common.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_copy_src_to_common.Transition.pResource = cudaTexture;
        barrier_copy_src_to_common.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier_copy_src_to_common.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        barrier_copy_src_to_common.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        State::state.commandList->ResourceBarrier(1, &barrier_copy_src_to_common);
        
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

        // Update window title
        char title[200];
        snprintf(title, sizeof(title),
                 "Ruben leip programma: %.1f ms - average: %.1f ms",
                 frameTimeMs, State::state.frameTimeAverager.getAverage());
        SetWindowTextA(State::state.hwnd, title);
    }
}
// This is the new, complete WndCreate function for main.cpp

void WndCreate(HWND hwnd)
{
#ifdef CONSOLE
    AllocConsole();
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    printf("Console Initialized.\n");
#endif

    // --- 1. Create Factory and Select Adapter ---
    UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            printf("D3D12 Debug Layer Enabled.\n");
        }
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)))) {
        fprintf(stderr, "Failed to create DXGI Factory.\n");
        return;
    }

    ComPtr<IDXGIAdapter1> hardwareAdapter;
    LUID adapterLUID = {};
    printf("Searching for a D3D12-capable hardware adapter...\n");
    for (UINT i = 0; factory->EnumAdapters1(i, &hardwareAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        hardwareAdapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        if (SUCCEEDED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
            adapterLUID = desc.AdapterLuid;
            printf("Found adapter: %S\n", desc.Description);
            break;
        }
    }
    if (!hardwareAdapter) {
        fprintf(stderr, "Failed to find a suitable D3D12 adapter.\n");
        return;
    }

    // --- 2. Create D3D12 Device ---
    if (FAILED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&State::state.d3dDevice)))) {
        fprintf(stderr, "Failed to create D3D12 device.\n");
        return;
    }
    printf("D3D12 Device created.\n");

    // --- 3. Link D3D12 Device with CUDA Context (CRITICAL STEP) ---
    int cudaDeviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&cudaDeviceCount);
    if (err != cudaSuccess) { fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err)); return; }

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
    if (cudaDevice == -1) { fprintf(stderr, "Could not find a matching CUDA device for the D3D12 adapter.\n"); return; }
    
    err = cudaSetDevice(cudaDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err)); return; }
    printf("CUDA device set successfully.\n");

    // --- 4. Create Core D3D12 Objects (Command Queue, Swap Chain, etc.) ---
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(State::state.d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&State::state.commandQueue)))) {
        fprintf(stderr, "Failed to create command queue.\n");
        return;
    }

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = g_frameCount;
    swapChainDesc.Width = State::dispWIDTH;
    swapChainDesc.Height = State::dispHEIGHT;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swapChain1;
    if (FAILED(factory->CreateSwapChainForHwnd(State::state.commandQueue.Get(), hwnd, &swapChainDesc, nullptr, nullptr, &swapChain1))) {
        fprintf(stderr, "Failed to create swap chain.\n");
        return;
    }
    if (FAILED(swapChain1.As(&State::state.swapChain))) {
        fprintf(stderr, "Failed to cast to IDXGISwapChain3.\n");
        return;
    }
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    State::state.frameIndex = State::state.swapChain->GetCurrentBackBufferIndex();

    // --- 5. Create Descriptor Heaps and Render Target Views ---
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = g_frameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(State::state.d3dDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&State::state.rtvHeap)))) {
        fprintf(stderr, "Failed to create RTV heap.\n");
        return;
    }
    State::state.rtvDescriptorSize = State::state.d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(State::state.rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < g_frameCount; n++) {
        State::state.swapChain->GetBuffer(n, IID_PPV_ARGS(&State::state.renderTargets[n]));
        State::state.d3dDevice->CreateRenderTargetView(State::state.renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, State::state.rtvDescriptorSize);
    }
    printf("Swap chain RTVs created.\n");

    // --- 6. Create Command Allocator, Command List, and Fence ---
    HRESULT hr = 0;
    for (UINT n = 0; n < g_frameCount; n++)
    {
        hr = State::state.d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&State::state.commandAllocators[n]));
        if (FAILED(hr)) { /* handle error */ }
    }
    hr = State::state.d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, State::state.commandAllocators[0].Get(), nullptr, IID_PPV_ARGS(&State::state.commandList));
    State::state.commandList->Close();
    State::state.d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&State::state.fence));
    State::state.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    for (UINT i = 0; i < g_frameCount; ++i) { State::state.fenceValues[i] = 0; }
    State::state.fenceValue = 1;

    // --- 7. Initialize CUDA-D3D Interop (This should now work) ---
    printf("Initializing CUDA-D3D12 Interop Framebuffer...\n");
    State::state.render->framebuffer.InitializeInterop(State::state.d3dDevice.Get(), State::dispWIDTH, State::dispHEIGHT);

    // --- 8. Initialize Rest of Application State ---
    printf("Initializing application state...\n");
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