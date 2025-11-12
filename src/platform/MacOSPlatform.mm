#include "platform/MacOSPlatform.hpp"
#import <Cocoa/Cocoa.h>

MacOSPlatform::MacOSPlatform(void* window, void* view) : _window(window), _view(view) {}

void MacOSPlatform::Create() {
    // This is called from macos_main after the window is created.
    // No extra work is needed here for now.
}

bool MacOSPlatform::IsKeyDown(char keycode) {
    // This is a placeholder. Real key handling would require an event monitor.
    return keysPressed.test(keycode);
}

// Implement the getter functions
void* MacOSPlatform::GetWindowHandle() {
    return _window;
}

void* MacOSPlatform::GetViewHandle() {
    return _view;
}