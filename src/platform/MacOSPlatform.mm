#include "platform/MacOSPlatform.hpp"
#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>  // key codes

MacOSPlatform::MacOSPlatform(void* window, void* view) : _window(window), _view(view) {}

void MacOSPlatform::Create() {
    // This is called from macos_main after the window is created.
    // No extra work is needed here for now.
}

bool MacOSPlatform::IsKeyDown(char keycode) {
    // Translate the platform-agnostic character into a platform-specific key code
    unsigned short nativeKeyCode;
    switch (keycode) {
        case 'W': nativeKeyCode = kVK_ANSI_W; break;
        case 'A': nativeKeyCode = kVK_ANSI_A; break;
        case 'S': nativeKeyCode = kVK_ANSI_S; break;
        case 'D': nativeKeyCode = kVK_ANSI_D; break;
        case 'Z': nativeKeyCode = kVK_ANSI_Z; break;
        case ' ': nativeKeyCode = kVK_Space;  break;
        default:
            nativeKeyCode = keycode;
            break;
    }
    
    // Now check the bitset using the correct NATIVE key code
    return keysPressed.test(nativeKeyCode);
}

// Implement the getter functions
void* MacOSPlatform::GetWindowHandle() {
    return _window;
}

void* MacOSPlatform::GetViewHandle() {
    return _view;
}