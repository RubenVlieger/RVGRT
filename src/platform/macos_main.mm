#define NS_AUTOMATED_REFCOUNT_ENABLED 1

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#include <memory>

#import "platform/GameView.h"
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
    GameView*  _view;
}
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    NSRect frame = NSMakeRect(0, 0, State::screenWIDTH, State::screenHEIGHT);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable;

    _window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:style
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [_window setTitle:@"RVGRT on Metal (Compute)"];
    [_window center];

    [_window setAcceptsMouseMovedEvents:YES];

    // 1. Create the Metal graphics device.
    State::state.graphicsDevice = std::make_unique<MetalDevice>();
    MetalDevice* metalDevice = static_cast<MetalDevice*>(State::state.graphicsDevice.get());

    id<MTLDevice> device = metalDevice->GetMetalDevice();
    // -----------------------

    // 2. Create the MetalKit View using the device.
    _view = [[GameView alloc] initWithFrame:frame device:device];   
    _view.delegate = self;
    _view.paused = YES;
    _view.enableSetNeedsDisplay = YES;

    // 3. Initialize the graphics device with the view handle.
    State::state.graphicsDevice->Initialize((__bridge void*)_view);

    // 4. Create the macOS platform abstraction.
    State::state.platform = std::make_unique<MacOSPlatform>((__bridge void*)_window, (__bridge void*)_view);

    // 5. Create the compute-based Metal renderer.
    State::state.renderer = std::make_unique<MetalRenderer>(device);

    _view.clearColor = MTLClearColorMake(0, 0, 0, 1);

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


- (void)gameLoop:(NSTimer *)timer {
    // Get the current time for calculating delta time
    static auto lastTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    double frameTimeMs = std::chrono::duration<double, std::milli>(currentTime - lastTime).count();
    std::cout << frameTimeMs << std::endl;
    lastTime = currentTime;

    State::state.platform->deltaTime = frameTimeMs;

    State::state.character.Update(0); 

    [_view setNeedsDisplay:YES];
}



- (void)drawInMTKView:(nonnull MTKView *)view 
{
    MetalRenderer* renderer = static_cast<MetalRenderer*>(State::state.renderer.get());
    MetalDevice* device = static_cast<MetalDevice*>(State::state.graphicsDevice.get());
    if (!renderer || !device) return;

    id<CAMetalDrawable> drawable = [view currentDrawable];
    if (!drawable) { return; }

    id<MTLCommandBuffer> commandBuffer = [device->GetMetalCommandQueue() commandBuffer];
    commandBuffer.label = @"FrameCommandBuffer";

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"MainComputeEncoder";
    renderer->Draw(encoder, State::state.character, 0);

    [encoder endEncoding];

    id<MTLTexture> sourceTexture = renderer->GetOutputTexture();
    id<MTLTexture> destinationTexture = drawable.texture;

    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    blitEncoder.label = @"BlitToScreen";

    [blitEncoder copyFromTexture:sourceTexture
                        sourceSlice:0 sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake(sourceTexture.width, sourceTexture.height, 1)
                        toTexture:destinationTexture
                    destinationSlice:0 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
    
    [blitEncoder endEncoding]; 

    [commandBuffer presentDrawable:drawable];

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