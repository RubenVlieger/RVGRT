#define NS_AUTOMATED_REFCOUNT_ENABLED 1

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#include <memory>

#include "State.hpp"
#include "platform/MacOSPlatform.hpp"
#include "renderer/MetalDevice.hpp"
#include "renderer/MetalRenderer.hpp"
#include "Character.hpp"

/**
 * @file macos_main.mm
 * @brief Entry point and main application loop for the macOS platform.
 *
 * This file sets up the native Cocoa application, window, and the MTKView for Metal rendering.
 * It follows a modern, event-driven approach:
 *
 * 1.  `applicationDidFinishLaunching:`: Initializes the window and all core engine components
 *     (Platform, GraphicsDevice, Renderer).
 *
 * 2.  `NSTimer`: A high-frequency timer is set up to call the `gameLoop` method. This creates
 *     a decoupled game loop for updating game logic (like character movement) at a consistent rate.
 *
 * 3.  `gameLoop`: Updates the character state and then calls `setNeedsDisplay:`, which tells the
 *     operating system that the view needs to be redrawn.
 *
 * 4.  `drawInMTKView:`: This is the OS-driven rendering callback. It's triggered by `setNeedsDisplay:`.
 *     Its sole responsibility is to produce a single frame. It orchestrates the two-step rendering process:
 *      a. Call `State::state.renderer->Draw(...)` which uses a compute shader to render the scene
 *         into an off-screen texture.
 *      b. Use a high-performance `MTLBlitCommandEncoder` to copy the renderer's finished texture
 *         to the view's drawable texture, which is then presented to the screen.
 */
@interface AppDelegate : NSObject <NSApplicationDelegate, MTKViewDelegate>
{
    NSWindow* _window;
    MTKView*  _view;
}
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    NSRect frame = NSMakeRect(0, 0, State::dispWIDTH, State::dispHEIGHT);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable;

    _window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:style
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [_window setTitle:@"RVGRT on Metal (Compute)"];
    [_window center];

    // --- Core Engine Initialization ---
    // 1. Create the Metal graphics device.
    State::state.graphicsDevice = std::make_unique<MetalDevice>();
    MetalDevice* metalDevice = static_cast<MetalDevice*>(State::state.graphicsDevice.get());

    // --- THE FIX IS HERE ---
    // We are calling a C++ method on a C++ object, so we must use C++ -> syntax.
    id<MTLDevice> device = metalDevice->GetMetalDevice();
    // -----------------------

    // 2. Create the MetalKit View using the device.
    _view = [[MTKView alloc] initWithFrame:frame device:device];
    _view.delegate = self;
    _view.paused = YES;
    _view.enableSetNeedsDisplay = YES;

    // 3. Initialize the graphics device with the view handle.
    // This is also a C++ method call on a C++ object.
    State::state.graphicsDevice->Initialize((__bridge void*)_view);

    // 4. Create the macOS platform abstraction.
    State::state.platform = std::make_unique<MacOSPlatform>((__bridge void*)_window, (__bridge void*)_view);

    // 5. Create the compute-based Metal renderer.
    State::state.renderer = std::make_unique<MetalRenderer>(device);

    // --- Final Window and App Setup ---
    [_window setContentView:_view];
    [_window makeFirstResponder:_view];
    [_window makeKeyAndOrderFront:nil];
    
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];

    // --- Start the Game Loop ---
    [NSTimer scheduledTimerWithTimeInterval:(1.0 / 60.0)
                                     target:self
                                   selector:@selector(gameLoop:)
                                   userInfo:nil
                                    repeats:YES];
}


// This is the main rendering callback. It only runs when the view is marked as "dirty".
- (void)drawInMTKView:(nonnull MTKView *)view {
    MetalRenderer* renderer = static_cast<MetalRenderer*>(State::state.renderer.get());
    MetalDevice* device = static_cast<MetalDevice*>(State::state.graphicsDevice.get());

    if (!renderer || !device) {
        return;
    }
    
    // STEP 1: Execute the compute shader to render the scene to an off-screen texture.
    renderer->Draw(State::state.character, 0); // Frame count can be passed in later

    // STEP 2: Copy the result to the screen.
    id<MTLTexture> sourceTexture = renderer->GetOutputTexture();
    
    // Get the texture provided by the system that will be displayed on the screen.
    id<CAMetalDrawable> drawable = [view currentDrawable];
    if (!drawable) { return; }
    id<MTLTexture> destinationTexture = drawable.texture;

    // Create a new command buffer specifically for our fast copy operation.
    id<MTLCommandBuffer> commandBuffer = [device->GetMetalCommandQueue() commandBuffer];
    commandBuffer.label = @"BlitToScreenBuffer";

    // Create a Blit Command Encoder, which is optimized for texture-to-texture copies.
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    blitEncoder.label = @"TextureBlit";
    
    // Schedule the copy from our rendered texture to the screen's texture.
    [blitEncoder copyFromTexture:sourceTexture
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(sourceTexture.width, sourceTexture.height, 1)
                        toTexture:destinationTexture
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];

    // We are done encoding copy commands.
    [blitEncoder endEncoding];

    // Schedule the presentation of the drawable after the copy command buffer has completed.
    [commandBuffer presentDrawable:drawable];

    // Commit the command buffer to the GPU to begin the copy and presentation.
    [commandBuffer commit];
}


// This delegate method is called when the user resizes the window.
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    MetalRenderer* renderer = static_cast<MetalRenderer*>(State::state.renderer.get());
    if (renderer) {
        renderer->OnResize(size.width, size.height);
    }
}


- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    // Cleanup will be handled by the unique_ptr destructors when State is destroyed.
    // Any manual cleanup would go here.
}

@end


// The standard C-style entry point for the application.
int MacOSMain(int argc, const char* argv[]) {
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        AppDelegate* delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}