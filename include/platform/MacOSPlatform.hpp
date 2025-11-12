#pragma once

#include "platform/Platform.hpp"

// Forward declare Objective-C types only when compiling Objective-C++ code
#ifdef __OBJC__
@class NSWindow;
@class NSView;
#endif

class MacOSPlatform : public Platform
{
public:
    MacOSPlatform(void* window, void* view);
    ~MacOSPlatform() override = default;

    void Create() override;
    bool IsKeyDown(char keycode) override;

    void* GetWindowHandle() override;
    void* GetViewHandle() override;

private:
    // Use void* in the header to remain C++ compatible
    void* _window;
    void* _view;
};