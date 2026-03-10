#ifdef WIN32

#include <atomic>
#include <thread>
#include <windows.h>

#include "State.hpp"
#include "renderer/CudaRender.cuh"
#include "platform/NetworkClient.hpp"
#include "platform/WindowsPlatform.hpp"
#include "renderer/D3D12Device.hpp"

// Forward declaration of the render loop
void renderLoop();
std::atomic<bool> running = true;

// The one and only main function for the entire project.
int main(int argc, char *argv[]) {
  // On Windows, call the Win32 entry point.
  return Win32Main(GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOW);
}

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

    // 3. Perform engine-specific (platform-agnostic) initialization
    State::state.renderer = std::make_unique<CudaRenderer>();

    State::state.networkClient = NetworkClient::Create();
    if (State::state.networkClient) {
      State::state.networkClient->Connect("ws://rvgrt.rubenvlieger.nl/ws");
    }

    // 4. Start the render thread
    std::thread renderThread(renderLoop);

    // 5. Run the message loop
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

void renderLoop() {
  // Get the concrete device type for D3D12-specific operations
  D3D12Device *d3d12Device =
      static_cast<D3D12Device *>(State::state.graphicsDevice.get());

  // The rest of this loop is nearly identical to your original,
  // just using the new device and platform objects.

  using clock = std::chrono::steady_clock;
  auto lastTime = clock::now();
  double frameTimeMs = 16.6f;
  unsigned int frameCount = 0;

  // --- Initialize DLSS resources once the device is created ---
  // (This part is moved from WndCreate to here, after device init)
  // The command list must be open to record commands for tagging.
  d3d12Device->BeginFrame(); // This will reset the allocator and command list
  ID3D12GraphicsCommandList *cmdList = d3d12Device->GetCommandList();
  d3d12Device->EndFrame();

  while (running) {
    State::state.graphicsDevice->BeginFrame();

    // The core rendering logic
    State::state.character.Update(frameCount);
    State::state.platform->deltaTime = (float)frameTimeMs / 1000.f;

    if (State::state.networkClient) {
      State::state.networkClient->SendState(State::state.character);
      State::state.networkClient->PollUpdates(State::state.otherCharacters);
    } else {
      static float npcTime = 0.0f;
      npcTime += frameTimeMs / 1000.0f;
      for (auto &npc : State::state.otherCharacters) {
        npc.UpdateTestNPC(npcTime, frameTimeMs / 1000.0f);
      }
    }

    if (State::state.renderer) {
      State::state.renderer->Draw(State::state.character, frameCount);
    }

    ID3D12Resource *backBuffer = d3d12Device->GetCurrentBackBuffer();
    ID3D12Resource *outputTexture = nullptr;
    if (State::state.renderer && State::state.renderer->GetOutputTexture()) {
        outputTexture = (ID3D12Resource*)State::state.renderer->GetOutputTexture();
    }

    if (outputTexture) {
      // Create explicit D3D12 barriers without d3dx12.h
      D3D12_RESOURCE_BARRIER copyBarriers[2] = {};
      copyBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      copyBarriers[0].Transition.pResource = outputTexture;
      copyBarriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      copyBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      copyBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

      copyBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      copyBarriers[1].Transition.pResource = backBuffer;
      copyBarriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      copyBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
      copyBarriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

      cmdList->ResourceBarrier(2, copyBarriers);
      cmdList->CopyResource(backBuffer, outputTexture);

      D3D12_RESOURCE_BARRIER finalBarriers[2] = {};
      finalBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      finalBarriers[0].Transition.pResource = outputTexture;
      finalBarriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      finalBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
      finalBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;

      finalBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      finalBarriers[1].Transition.pResource = backBuffer;
      finalBarriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      finalBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      finalBarriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;

      cmdList->ResourceBarrier(2, finalBarriers);
    }

    State::state.graphicsDevice->EndFrame();

    // --- Frame Timing ---
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
}

#endif // WIN32