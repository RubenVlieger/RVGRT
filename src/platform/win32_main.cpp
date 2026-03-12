#ifdef WIN32

#include <atomic>
#include <thread>
#include <windows.h>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#include "State.hpp"
#include "platform/NetworkClient.hpp"
#include "platform/WindowsPlatform.hpp"
#include "renderer/D3D12/D3D12Device.hpp"
#include "renderer/CUDA/CudaRender.cuh"
#include "CudaD3D12Texture.cuh"
#include <d3d12/d3dx12.h>

#ifdef HAS_STREAMLINE
#include <sl.h>
#include <sl_dlss.h>
#endif

// Forward declarations
void renderLoop();
int WINAPI Win32Main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow);
std::atomic<bool> running = true;



int WINAPI Win32Main(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                     LPSTR lpCmdLine, int nCmdShow) {
  try {
    // 1. Create the platform-specific window and input handler
    State::state.platform = std::make_unique<WindowsPlatform>();
    State::state.platform->Create(); // This creates the HWND

    // 2. Create the platform-specific graphics device
    State::state.graphicsDevice = std::make_unique<D3D12Device>();
    State::state.graphicsDevice->Initialize(
        State::state.platform->GetWindowHandle());

    // 3. Create the CUDA renderer (mirrors macOS MetalRenderer creation)
    State::state.renderer = std::make_unique<CudaRenderer>();
    
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "CUDA error after init: %s\n", cudaGetErrorString(syncErr));
        fflush(stderr);
    }
    
    // Initialize DLSS if available
    D3D12Device* d3d12Device = static_cast<D3D12Device*>(State::state.graphicsDevice.get());
    CudaRenderer* cudaRenderer = static_cast<CudaRenderer*>(State::state.renderer.get());
    cudaRenderer->InitializeDLSS(d3d12Device->GetD3D12Device(), 
                                  State::dispWIDTH, State::dispHEIGHT);

    // 4. Network client setup
    State::state.networkClient = NetworkClient::Create();
    if (State::state.networkClient) {
      State::state.networkClient->Connect("ws://rvgrt.rubenvlieger.nl/ws");
    }

    // 5. Start the render thread
    std::thread renderThread(renderLoop);

    // 6. Run the message loop
    MSG msg = {};
    while (running) {
      if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
          running = false;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }
    }

    // Wait for the render thread to finish
    renderThread.join();

    return (int)msg.wParam;

  } catch (const std::runtime_error &e) {
    MessageBoxA(NULL, e.what(), "Initialization Error", MB_OK | MB_ICONERROR);
    return -1;
  }
}

// Streamline helper functions
#ifdef HAS_STREAMLINE
static inline void successCheck(sl::Result result, const char* msg) {
  if(result != sl::Result::eOk) {
    std::cerr << "sl error (" << msg << "): " << (int)result << "\n";
    throw std::runtime_error(msg);
  }
}

// Constants for DLSS
static constexpr sl::DLSSOptions kDlssOptions = {
  sl::DLSSMode::eMaxPerformance,
  sl::DLSSPreset::eDefault,
  sl::DLSSPreset::eDefault,
  0.0f, 0.0f,
  0.0f,
  0.0f, 0.0f,
  false
};
#endif

void renderLoop() {
  try {
    printf("[Render] Thread started\n"); fflush(stdout);

    D3D12Device *d3d12Device =
        static_cast<D3D12Device *>(State::state.graphicsDevice.get());
    CudaRenderer *cudaRenderer =
        static_cast<CudaRenderer *>(State::state.renderer.get());
    
    // Create interop texture for CUDA -> D3D12 output
    CudaD3D12Texture interopTexture;
    printf("[Render] Initializing interop texture\n"); fflush(stdout);
    interopTexture.Initialize(
        d3d12Device->GetD3D12Device(),
        State::dispWIDTH,
        State::dispHEIGHT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        L"CUDA-D3D12 Interop"
    );
    printf("[Render] Interop texture initialized\n"); fflush(stdout);

    using clock = std::chrono::steady_clock;
    auto lastTime = clock::now();
    double frameTimeMs = 16.6f;
    unsigned int frameCount = 0;
    float npcTime = 0.0f;

    while (running) {
      printf("[Render] BeginFrame\n"); fflush(stdout);
      State::state.graphicsDevice->BeginFrame();

      // Update game state
      State::state.character.Update(frameCount);
      State::state.platform->deltaTime = (float)frameTimeMs / 1000.f;

      // Network updates
      if (State::state.networkClient) {
        State::state.networkClient->SendState(State::state.character);
        State::state.networkClient->PollUpdates(State::state.otherCharacters);
      } else {
        npcTime += frameTimeMs / 1000.0f;
        for (auto &npc : State::state.otherCharacters) {
          npc.UpdateTestNPC(npcTime, frameTimeMs / 1000.0f);
        }
      }

      // MAIN RENDER CALL
      if (State::state.renderer) {
        printf("[Render] About to Draw\n"); fflush(stdout);
        cudaRenderer->Draw(State::state.character, frameCount);
        printf("[Render] Draw complete\n"); fflush(stdout);
      }
      
      printf("[Render] PostDraw and resource barriers\n"); fflush(stdout);
      // Copy CUDA output to D3D12 interop texture
      cudaRenderer->PostDraw(interopTexture.getCudaSurfObject(), 
                              State::dispWIDTH, State::dispHEIGHT, false);

      // Copy to backbuffer
      ID3D12Resource *backBuffer = d3d12Device->GetCurrentBackBuffer();
      ID3D12GraphicsCommandList *cmdList = d3d12Device->GetCommandList();
      ID3D12Resource *outputResource = interopTexture.GetD3D12Resource();

      D3D12_RESOURCE_BARRIER copyBarriers[] = {
          CD3DX12_RESOURCE_BARRIER::Transition(
              outputResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              D3D12_RESOURCE_STATE_COPY_SOURCE),
          CD3DX12_RESOURCE_BARRIER::Transition(backBuffer,
                                               D3D12_RESOURCE_STATE_PRESENT,
                                               D3D12_RESOURCE_STATE_COPY_DEST)};
      cmdList->ResourceBarrier(_countof(copyBarriers), copyBarriers);
      cmdList->CopyResource(backBuffer, outputResource);

      D3D12_RESOURCE_BARRIER finalBarriers[] = {
          CD3DX12_RESOURCE_BARRIER::Transition(outputResource,
                                               D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_COMMON),
          CD3DX12_RESOURCE_BARRIER::Transition(backBuffer,
                                               D3D12_RESOURCE_STATE_COPY_DEST,
                                               D3D12_RESOURCE_STATE_PRESENT)};
      cmdList->ResourceBarrier(_countof(finalBarriers), finalBarriers);

      printf("[Render] EndFrame\n"); fflush(stdout);
      State::state.graphicsDevice->EndFrame();

      // Frame Timing
      frameTimeMs =
          std::chrono::duration<double, std::milli>(clock::now() - lastTime)
              .count();
      State::state.platform->frameTimeAverager.addFrameTime(frameTimeMs);
      lastTime = clock::now();

      // Update window title
      char title[256];
      snprintf(title, sizeof(title), "RVGRT: %.1f ms | Avg: %.1f ms", frameTimeMs,
               State::state.platform->frameTimeAverager.getAverage());
      SetWindowTextA(static_cast<HWND>(State::state.platform->GetWindowHandle()),
                     title);
      frameCount++;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "Render thread crashed with exception: %s\n", e.what());
    fflush(stderr);
    running = false;
  }
}

#endif // WIN32
