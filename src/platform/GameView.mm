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
    
    // Set the corresponding bit in our cross-platform bitset
    platform->keysPressed.set(event.keyCode, 1);
    
    // Add a key to toggle mouse lock (Escape key)
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

@end