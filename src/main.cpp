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