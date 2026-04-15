// src/platform/GameView.mm
#import "platform/GameView.h"
#include "platform/MacOSPlatform.hpp"
#include "State.hpp"
#include <IOKit/hidsystem/ev_keymap.h> // For virtual key codes
#import <Carbon/Carbon.h>  // key codes


@implementation GameView
{
    BOOL _mouseIsLocked;
}

// Override the initializer
- (instancetype)initWithFrame:(NSRect)frameRect device:(id<MTLDevice>)device {
    self = [super initWithFrame:frameRect device:device];
    if (self) {
        // Start with the mouse locked
        [self setMouseLock:YES];
    }
    return self;
}

// We need to tell the system that our view can become the primary focus for events.
- (BOOL)acceptsFirstResponder {
    return YES;
}

// This method is called whenever a key is pressed down.
- (void)keyDown:(NSEvent *)event {
    // Get the platform object from our global state
    Platform* platform = State::state.platform.get();
    if (!platform || [event isARepeat]) {
        return;
    }

    // Console is open — route all input to the console
    if (State::state.console.IsOpen()) {
        if (event.keyCode == kVK_Escape) {
            platform->keysPressed.set(event.keyCode, 1);
        } else if (event.keyCode == kVK_Return) {
            platform->keysPressed.set(event.keyCode, 1);
        } else if (event.keyCode == kVK_Delete) {
            platform->keysPressed.set(event.keyCode, 1);
        } else if (event.keyCode == kVK_UpArrow) {
            platform->keysPressed.set(event.keyCode, 1);
        } else if (event.keyCode == kVK_DownArrow) {
            platform->keysPressed.set(event.keyCode, 1);
        } else {
            // Printable ASCII characters
            NSString* chars = [event characters];
            if (chars.length > 0) {
                unichar c = [chars characterAtIndex:0];
                if (c >= 32 && c < 127) {
                    std::lock_guard<std::mutex> lock(platform->textInputMutex);
                    platform->textInputQueue.push(static_cast<char>(c));
                }
            }
        }
        return;
    }

    // Console toggle keys — 'T' opens blank, '/' opens with prefix
    if (event.keyCode == kVK_ANSI_T || event.keyCode == kVK_ANSI_Slash) {
        char prefix = (event.keyCode == kVK_ANSI_Slash) ? '/' : 0;
        State::state.console.Open(prefix);
        platform->consoleOpen = true;
        [self setMouseLock:NO];
        return;
    }

    // Set the corresponding bit in our cross-platform bitset
    platform->keysPressed.set(event.keyCode, 1);

    // Escape key toggles mouse lock when console is not open
    if (event.keyCode == kVK_Escape) {
        [self setMouseLock:!_mouseIsLocked];
    }
}

// This method is called whenever a key is released.
- (void)keyUp:(NSEvent *)event {
    Platform* platform = State::state.platform.get();
    if (!platform) {
        return;
    }
    // Clear the bit
    platform->keysPressed.set(event.keyCode, 0);
}

// This method is called whenever the mouse moves.
- (void)mouseMoved:(NSEvent *)event {
    Platform* platform = State::state.platform.get();
    if (!platform || !_mouseIsLocked) {
        return;
    }
    
    // NSEvent provides the change in position (delta), which is exactly what we need.
    long deltaX = [event deltaX];
    long deltaY = [event deltaY];

    // Update the atomic delta values that the Character class reads
    platform->deltaXMouse.fetch_add(deltaX, std::memory_order_relaxed);
    platform->deltaYMouse.fetch_add(deltaY, std::memory_order_relaxed);
}

// We need to handle mouse drags as well for continuous input
- (void)mouseDragged:(NSEvent *)event { [self mouseMoved:event]; }
- (void)rightMouseDragged:(NSEvent *)event { [self mouseMoved:event]; }
- (void)otherMouseDragged:(NSEvent *)event { [self mouseMoved:event]; }

// Block interaction: left click removes block, right click places block
- (void)mouseDown:(NSEvent *)event {
    Platform* platform = State::state.platform.get();
    if (!platform) return;
    platform->leftMouseJustPressed.store(true, std::memory_order_relaxed);
}

- (void)rightMouseDown:(NSEvent *)event {
    Platform* platform = State::state.platform.get();
    if (!platform) return;
    platform->rightMouseJustPressed.store(true, std::memory_order_relaxed);
}


// A helper function to control the cursor's visibility and behavior
- (void)setMouseLock:(BOOL)locked {
    _mouseIsLocked = locked;
    
    if (locked) {
        // Hide the cursor and lock it to the view
        [NSCursor hide];
        CGAssociateMouseAndMouseCursorPosition(false);
    } else {
        // Show the cursor and let it move freely
        [NSCursor unhide];
        CGAssociateMouseAndMouseCursorPosition(true);
    }
}

- (void)flagsChanged:(NSEvent *)event {
    Platform* platform = State::state.platform.get();
    if (!platform) return;

    // 0x38 is kVK_Shift (Left Shift)
    // Check if the Shift flag is currently active in the event
    bool isShiftDown = (event.modifierFlags & NSEventModifierFlagShift) != 0;
    
    // 0x38 is the virtual key code for Shift
    if (isShiftDown) {
        platform->keysPressed.set(0x38, 1);
    } else {
        platform->keysPressed.set(0x38, 0);
    }
}

@end