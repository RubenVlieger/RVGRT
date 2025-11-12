#pragma once

#ifdef _WIN32
#include "platform/Platform.hpp"
#include <windows.h> // For HWND

class WindowsPlatform : public Platform
{
public:
    WindowsPlatform();
    ~WindowsPlatform() override = default;

    // Initializes the platform, including creating the window and setting up raw input.
    void Create() override;
    
    // Checks the internal key state bitset.
    bool IsKeyDown(char keycode) override;

    // Returns the native window handles.
    void* GetWindowHandle() override { return hwnd; }
    void* GetViewHandle() override { return hwnd; } // For Win32, the view and window are the same handle.

    HWND hwnd = nullptr;

private:
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void RegisterWindowClass();
};

#endif // _WIN32