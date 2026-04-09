#ifdef _WIN32

#include "platform/WindowsPlatform.hpp"
#include "State.hpp"
#include <hidsdi.h> // For RAWINPUTDEVICE
#include <windowsx.h>

const char g_szClassName[] = "RVGRTWindowClass";

WindowsPlatform::WindowsPlatform() {}

LRESULT CALLBACK WindowsPlatform::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // A static WndProc that routes messages to the correct instance.
    WindowsPlatform* platform = nullptr;
    if (msg == WM_CREATE) {
        CREATESTRUCT* pCreate = (CREATESTRUCT*)lParam;
        platform = (WindowsPlatform*)pCreate->lpCreateParams;
        SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)platform);
    } else {
        platform = (WindowsPlatform*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    }

    if (platform) {
        switch(msg) {
            case WM_KEYDOWN:
                // Console toggle: T or '/' opens console
                if (State::state.console.IsOpen()) {
                    if (wParam == VK_RETURN) {
                        platform->keysPressed.set(VK_RETURN, 1);
                    } else if (wParam == VK_BACK) {
                        platform->keysPressed.set(VK_BACK, 1);
                    } else if (wParam == VK_UP) {
                        platform->keysPressed.set(VK_UP, 1);
                    } else if (wParam == VK_DOWN) {
                        platform->keysPressed.set(VK_DOWN, 1);
                    } else if (wParam == VK_ESCAPE) {
                        platform->keysPressed.set(VK_ESCAPE, 1);
                    }
                    return 0;
                }
                if (wParam == 0x54 || wParam == VK_OEM_2) { // 'T' or '/'
                    char prefix = (wParam == VK_OEM_2) ? '/' : 0;
                    State::state.console.Open(prefix);
                    platform->consoleOpen = true;
                    return 0;
                }
                platform->keysPressed.set((unsigned long)wParam, 1);
                if (wParam == VK_ESCAPE) {
                    PostQuitMessage(0);
                }
                return 0;

            case WM_CHAR:
                if (State::state.console.IsOpen()) {
                    if (wParam >= 32 && wParam < 127) {
                        std::lock_guard<std::mutex> lock(platform->textInputMutex);
                        platform->textInputQueue.push(static_cast<char>(wParam));
                    }
                    return 0;
                }
                return 0;

            case WM_KEYUP:
                platform->keysPressed.set((unsigned char)wParam, 0);
                return 0;

            case WM_INPUT: {
                UINT size;
                GetRawInputData((HRAWINPUT)lParam, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER));
                static std::vector<BYTE> rawBuffer(size);
                if (rawBuffer.size() < size) rawBuffer.resize(size);

                GetRawInputData((HRAWINPUT)lParam, RID_INPUT, rawBuffer.data(), &size, sizeof(RAWINPUTHEADER));
                RAWINPUT* raw = (RAWINPUT*)rawBuffer.data();
                if (raw->header.dwType == RIM_TYPEMOUSE) {
                    platform->deltaXMouse.fetch_add(raw->data.mouse.lLastX, std::memory_order_relaxed);
                    platform->deltaYMouse.fetch_add(raw->data.mouse.lLastY, std::memory_order_relaxed);
                }
                return 0;
            }

            case WM_CLOSE:
                DestroyWindow(hwnd);
                return 0;

            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void WindowsPlatform::RegisterWindowClass() {
    WNDCLASSEX wc = {};
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WindowsPlatform::WndProc;
    wc.hInstance     = GetModuleHandle(NULL);
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = g_szClassName;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&wc)) {
        throw std::runtime_error("Window Registration Failed!");
    }
}

void WindowsPlatform::Create() {
    RegisterWindowClass();

    hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE, g_szClassName, "RVGRT - Voxel World Engine",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, State::screenWIDTH, State::screenHEIGHT,
        NULL, NULL, GetModuleHandle(NULL), this // Pass 'this' to WM_CREATE
    );

    if (!hwnd) {
        throw std::runtime_error("Window Creation Failed!");
    }

    // Register for Raw Input for smooth mouse movement
    RAWINPUTDEVICE rid[1] = {};
    rid[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
    rid[0].usUsage     = HID_USAGE_GENERIC_MOUSE;
    rid[0].dwFlags     = RIDEV_INPUTSINK | RIDEV_NOLEGACY;
    rid[0].hwndTarget  = hwnd;
    if (!RegisterRawInputDevices(rid, 1, sizeof(rid[0]))) {
         throw std::runtime_error("Failed to register raw input device.");
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
}

bool WindowsPlatform::IsKeyDown(char keycode) {
    return keysPressed.test(keycode);
}

#endif // _WIN32