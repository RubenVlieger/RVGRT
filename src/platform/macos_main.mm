#define NS_AUTOMATED_REFCOUNT_ENABLED 1

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#import <Carbon/Carbon.h>
#include <memory>
#include <string>
#include <ctime>
#include <cmath>

#include "Character.hpp"
#include "State.hpp"
#include "VoxelQuery.hpp"
#include "console/GameConsole.hpp"
#import "platform/GameView.h"
#include "platform/MacOSPlatform.hpp"
#include "platform/NetworkClient.hpp"
#include "renderer/Metal/MetalDevice.hpp"
#include "renderer/Metal/MetalRenderer.hpp"

// Global block edit overlay map (defined in VoxelQuery.hpp as extern)
std::unordered_map<glm::ivec3, uint8_t> g_blockEdits;

/**
 * @file macos_main.mm
 * @brief Entry point and main application loop for the macOS platform.
 *
 * This file sets up the native Cocoa application, window, and the MTKView for
 * Metal rendering. It follows a modern, event-driven approach:
 *
 * 1.  `applicationDidFinishLaunching:`: Initializes the window and all core
 * engine components (Platform, GraphicsDevice, Renderer).
 *
 * 2.  `NSTimer`: A high-frequency timer is set up to call the `gameLoop`
 * method. This creates a decoupled game loop for updating game logic (like
 * character movement) at a consistent rate.
 *
 * 3.  `gameLoop`: Updates the character state and then calls
 * `setNeedsDisplay:`, which tells the operating system that the view needs to
 * be redrawn.
 *
 * 4.  `drawInMTKView:`: This is the OS-driven rendering callback. It's
 * triggered by `setNeedsDisplay:`. Its sole responsibility is to produce a
 * single frame. It orchestrates the two-step rendering process: a. Call
 * `State::state.renderer->Draw(...)` which uses a compute shader to render the
 * scene into an off-screen texture. b. Use a high-performance
 * `MTLBlitCommandEncoder` to copy the renderer's finished texture to the view's
 * drawable texture, which is then presented to the screen.
 */
@interface AppDelegate : NSObject <NSApplicationDelegate, MTKViewDelegate> {
  NSWindow *_window;
  GameView *_view;
}
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
  CGFloat scaleFactor = [[NSScreen mainScreen] backingScaleFactor];
  NSRect frame = NSMakeRect(0, 0, State::screenWIDTH / scaleFactor, State::screenHEIGHT / scaleFactor);
  NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable;

  _window = [[NSWindow alloc] initWithContentRect:frame
                                        styleMask:style
                                          backing:NSBackingStoreBuffered
                                            defer:NO];
  [_window setTitle:@"RVGRT on Metal (Compute)"];
  [_window center];

  [_window setAcceptsMouseMovedEvents:YES];

  // 1. Create the Metal graphics device.
  State::state.graphicsDevice = std::make_unique<MetalDevice>();
  MetalDevice *metalDevice =
      static_cast<MetalDevice *>(State::state.graphicsDevice.get());

  id<MTLDevice> device = metalDevice->GetMetalDevice();
  // -----------------------

  // 2. Create the MetalKit View using the device.
  _view = [[GameView alloc] initWithFrame:frame device:device];
  _view.delegate = self;
  _view.paused = YES;
  _view.enableSetNeedsDisplay = YES;

  // 3. Initialize the graphics device with the view handle.
  State::state.graphicsDevice->Initialize((__bridge void *)_view);

  // 4. Create the macOS platform abstraction.
  State::state.platform = std::make_unique<MacOSPlatform>(
      (__bridge void *)_window, (__bridge void *)_view);

  State::state.networkClient = NetworkClient::Create();
  State::state.networkClient->Connect("wss://rvgrt.rubenvlieger.nl/ws");
  State::state.networkClient->SetChatCallback([](int clientId, const std::string& sender, const std::string& text) {
    State::state.console.OnChatReceived(clientId, sender, text);
  });
  State::state.console.SetChatSendCallback([](const std::string& sender, const std::string& text) {
    State::state.networkClient->SendChat(sender, text);
  });

  // ── Block Network Sync Callbacks (Phase 4) ─────────────────────────────────
  // Remote block edits from other players are queued on the WebSocket thread
  // and applied on the main thread during game loop.
  State::state.networkClient->SetBlockEditCallback([](int32_t x, int32_t y, int32_t z, uint8_t matID) {
    std::lock_guard<std::mutex> lock(State::state.blockEditsMutex);
    State::state.pendingRemoteEdits.push_back({x, y, z, matID});
  });
  State::state.networkClient->SetBlockSyncCallback([](const std::vector<BlockEdit>& edits) {
    std::lock_guard<std::mutex> lock(State::state.blockEditsMutex);
    State::state.pendingRemoteEdits = edits;
  });
  State::state.networkClient->SetBlockResetCallback([]() {
    State::state.blockResetRequested = true;
  });

  // 5. Create the compute-based Metal renderer.
  State::state.renderer = std::make_unique<MetalRenderer>(device);

  // 6. Initialize the in-game console.
  State::state.console.Initialize();

  _view.clearColor = MTLClearColorMake(0, 0, 0, 1);

  // --- Final Window and App Setup ---
  [_window setContentView:_view];
  [_window makeFirstResponder:_view];
  [_window makeKeyAndOrderFront:nil];
  _view.autoResizeDrawable = NO;

  _view.drawableSize = CGSizeMake(State::screenWIDTH, State::screenHEIGHT);
  _view.layer.contentsGravity = kCAGravityResizeAspectFill;

  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  [NSApp activateIgnoringOtherApps:YES];

  // --- Start the Game Loop ---
  NSTimer* timer = [NSTimer timerWithTimeInterval:(1.0 / 60.0)
                                           target:self
                                         selector:@selector(gameLoop:)
                                         userInfo:nil
                                          repeats:YES];
  [[NSRunLoop mainRunLoop] addTimer:timer forMode:NSRunLoopCommonModes];
}

- (void)gameLoop:(NSTimer *)timer {
  static unsigned int globalFrameCount = 0;

  // Get the current time for calculating delta time
  static auto lastTime = std::chrono::steady_clock::now();
  auto currentTime = std::chrono::steady_clock::now();
  double frameTimeMs =
      std::chrono::duration<double, std::milli>(currentTime - lastTime).count();
  lastTime = currentTime;

  State::state.platform->deltaTime = frameTimeMs;

  State::state.console.Update(frameTimeMs / 1000.0f);

  // Drain console text input queue
  if (State::state.console.IsOpen()) {
    Platform* platform = State::state.platform.get();
    if (platform) {
      std::lock_guard<std::mutex> lock(platform->textInputMutex);
      while (!platform->textInputQueue.empty()) {
        char c = platform->textInputQueue.front();
        platform->textInputQueue.pop();
        State::state.console.OnCharInput(c);
      }

      // Handle special keys for console
      if (platform->IsKeyDown(kVK_Return)) {
        State::state.console.OnSpecialKey(SpecialKey::Enter);
        platform->keysPressed.reset(kVK_Return);
      }
      if (platform->IsKeyDown(kVK_Delete)) {
        State::state.console.OnSpecialKey(SpecialKey::Backspace);
        platform->keysPressed.reset(kVK_Delete);
      }
      if (platform->IsKeyDown(kVK_UpArrow)) {
        State::state.console.OnSpecialKey(SpecialKey::ArrowUp);
        platform->keysPressed.reset(kVK_UpArrow);
      }
      if (platform->IsKeyDown(kVK_DownArrow)) {
        State::state.console.OnSpecialKey(SpecialKey::ArrowDown);
        platform->keysPressed.reset(kVK_DownArrow);
      }
      if (platform->IsKeyDown(kVK_Escape)) {
        State::state.console.OnSpecialKey(SpecialKey::Escape);
        platform->keysPressed.reset(kVK_Escape);
        // Console was closed — re-lock mouse
        [_view setMouseLock:YES];
        platform->consoleOpen = false;
      }
    }
  }

  // Only update character movement when console is closed
  if (!State::state.console.IsOpen()) {
    State::state.character.Update(globalFrameCount);
  }

  // ── Block Interaction (Phase 2) ──────────────────────────────────────────
  // Left click removes block, right click places block at crosshair.
  // Only process clicks when console is closed and mouse is locked.
  if (!State::state.console.IsOpen()) {
    Platform* platform = State::state.platform.get();
    if (platform) {
      if (platform->leftMouseJustPressed.exchange(false)) {
        auto& c = State::state.character;
        glm::vec3 eyePos(c.position.x, c.position.y, c.position.z);
        glm::vec3 dir(static_cast<float>(c.direction.x),
                      static_cast<float>(c.direction.y),
                      static_cast<float>(c.direction.z));
        RaycastResult hit = RaycastDDA(eyePos, dir, 8.0f);
        if (hit.hit) {
          auto& mm = static_cast<MetalRenderer*>(State::state.renderer.get())->GetMaterialMap();
          if (mm.RemoveVoxel(hit.voxelX, hit.voxelY, hit.voxelZ)) {
            State::state.localBlockEdits.push_back({hit.voxelX, hit.voxelY, hit.voxelZ, 0});
            State::state.networkClient->SendBlockEdit(hit.voxelX, hit.voxelY, hit.voxelZ, 0);
          }
        }
      }
      if (platform->rightMouseJustPressed.exchange(false)) {
        auto& c = State::state.character;
        glm::vec3 eyePos(c.position.x, c.position.y, c.position.z);
        glm::vec3 dir(static_cast<float>(c.direction.x),
                      static_cast<float>(c.direction.y),
                      static_cast<float>(c.direction.z));
        RaycastResult hit = RaycastDDA(eyePos, dir, 8.0f);
        if (hit.hit) {
          auto& mm = static_cast<MetalRenderer*>(State::state.renderer.get())->GetMaterialMap();
          if (mm.PlaceVoxel(hit.adjacentX, hit.adjacentY, hit.adjacentZ,
                             State::state.selectedMaterialID)) {
            State::state.localBlockEdits.push_back(
                {hit.adjacentX, hit.adjacentY, hit.adjacentZ,
                 State::state.selectedMaterialID});
            State::state.networkClient->SendBlockEdit(hit.adjacentX, hit.adjacentY, hit.adjacentZ,
                                                       State::state.selectedMaterialID);
          }
        }
      }
    }
  }

  // ── Remote Block Edit Queue (Phase 4) ────────────────────────────────
  // Drain block edits received from other players via the network.
  // These are queued on the WebSocket thread and applied here on the main
  // thread where MaterialMap is accessible.
  {
    std::lock_guard<std::mutex> lock(State::state.blockEditsMutex);
    if (!State::state.pendingRemoteEdits.empty()) {
      auto& mm = static_cast<MetalRenderer*>(State::state.renderer.get())->GetMaterialMap();
      for (auto& edit : State::state.pendingRemoteEdits) {
        if (edit.matID == 0)
          mm.RemoveVoxel(edit.x, edit.y, edit.z);
        else
          mm.PlaceVoxel(edit.x, edit.y, edit.z, edit.matID);
        SetBlockEdit(edit.x, edit.y, edit.z, edit.matID);
      }
      State::state.pendingRemoteEdits.clear();
    }
  }

  // ── Block Reset Handling ──────────────────────────────────────────────
  // Processed here because MaterialMap is only accessible from platform code.
  if (State::state.blockResetRequested) {
    auto& mm = static_cast<MetalRenderer*>(State::state.renderer.get())->GetMaterialMap();
    mm.ResetBlockEdits();
    State::state.localBlockEdits.clear();
    State::state.blockResetRequested = false;
  }

  if (State::state.networkClient) {
    State::state.networkClient->SendState(State::state.character);
    State::state.networkClient->PollUpdates(State::state.otherCharacters);
  } else {
    // Fallback or offline NPC update
    static float npcTime = 0.0f;
    npcTime += frameTimeMs / 1000.0f;
    for (auto &npc : State::state.otherCharacters) {
      npc.UpdateTestNPC(npcTime, frameTimeMs / 1000.0f);
    }
  }

  globalFrameCount++;

  [_view setNeedsDisplay:YES];
}

- (void)drawInMTKView:(nonnull MTKView *)view {
  MetalRenderer *renderer =
      static_cast<MetalRenderer *>(State::state.renderer.get());
  MetalDevice *device =
      static_cast<MetalDevice *>(State::state.graphicsDevice.get());

  if (!renderer || !device)
    return;

  id<CAMetalDrawable> drawable = [view currentDrawable];
  if (!drawable)
    return;

  id<MTLCommandBuffer> commandBuffer =
      [device->GetMetalCommandQueue() commandBuffer];
  commandBuffer.label = @"FrameCommandBuffer";

  renderer->Draw(commandBuffer, State::state.character, 0);

  id<MTLCounterSampleBuffer> counterBuf =
      (id<MTLCounterSampleBuffer>)renderer->GetCounterBuffer();
  id<MTLBuffer> timestampBuf = (id<MTLBuffer>)renderer->GetTimestampBuffer();

  if (counterBuf && timestampBuf) {
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    blit.label = @"ResolveTimestamps";

    [blit resolveCounters:counterBuf
                  inRange:NSMakeRange(0, 20)
        destinationBuffer:timestampBuf
        destinationOffset:0];

    [blit endEncoding];
  }

  id<MTLTexture> sourceTexture = renderer->GetOutputTexture();
  id<MTLTexture> destinationTexture = drawable.texture;

  id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
  blitEncoder.label = @"BlitToScreen";

  [blitEncoder copyFromTexture:sourceTexture
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(sourceTexture.width,
                                           sourceTexture.height, 1)
                     toTexture:destinationTexture
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];

  [blitEncoder endEncoding];

  // Benchmark data collection - static to persist across frames
  static struct {
    int frameCount = 0;
    double data[50][11]; // 50 frames, 11 timing values per frame
    double cpuTextPrep[50];
    double cpuStreaming[50];
    double cpuDrawTotal[50];
  } benchmark;

  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    if (cb.status != MTLCommandBufferStatusCompleted)
      return;

    double msApprox = 0.0, msGBuffer = 0.0, msIndirect = 0.0;
    double msAccum = 0.0, msDenoise = 0.0, msComp = 0.0;
    double msFX = 0.0, msFog = 0.0, msExpos = 0.0, msTextOverlay = 0.0;

    if (timestampBuf) {
      uint64_t *stamps = (uint64_t *)timestampBuf.contents;

      auto calcMs = [&](int start, int end) {
        if (stamps[end] < stamps[start])
          return 0.0;
        return (double)(stamps[end] - stamps[start]) / 1000000.0;
      };

      msApprox = calcMs(0, 1);
      msGBuffer = calcMs(2, 3);
      msIndirect = calcMs(4, 5);
      msAccum = calcMs(6, 7);
      msDenoise = calcMs(8, 9);
      msFog = calcMs(10, 11);
      msExpos = calcMs(12, 13);
      msComp = calcMs(14, 15);
      msTextOverlay = calcMs(16, 17);
      msFX = calcMs(18, 19);
    }

    // Store frame data for benchmark
    if (benchmark.frameCount < 50) {
      benchmark.data[benchmark.frameCount][0] = msApprox;
      benchmark.data[benchmark.frameCount][1] = msGBuffer;
      benchmark.data[benchmark.frameCount][2] = msIndirect;
      benchmark.data[benchmark.frameCount][3] = msAccum;
      benchmark.data[benchmark.frameCount][4] = msDenoise;
      benchmark.data[benchmark.frameCount][5] = msFog;
      benchmark.data[benchmark.frameCount][6] = msExpos;
      benchmark.data[benchmark.frameCount][7] = msComp;
      benchmark.data[benchmark.frameCount][8] = msTextOverlay;
      benchmark.data[benchmark.frameCount][9] = msFX;
      benchmark.data[benchmark.frameCount][10] = msApprox + msGBuffer + msIndirect + msAccum + msDenoise + msFog + msExpos + msComp + msTextOverlay + msFX;
      benchmark.cpuTextPrep[benchmark.frameCount] = renderer->cpuTextPrepMs;
      benchmark.cpuStreaming[benchmark.frameCount] = renderer->cpuStreamingMs;
      benchmark.cpuDrawTotal[benchmark.frameCount] = renderer->cpuDrawTotalMs;
      benchmark.frameCount++;

      // Write benchmark.log after 50 frames
      if (benchmark.frameCount == 50) {
        FILE* log = fopen("/Users/rubenvlieger/Documents/RVGRT/benchmark.log", "w");
        if (log) {
          // Header
          time_t now = time(nullptr);
          struct tm* timeinfo = localtime(&now);
          char timestr[64];
          strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", timeinfo);

          fprintf(log, "═══════════════════════════════════════════════════════════════\n");
          fprintf(log, "RVGRT Benchmark Report — %s\n", timestr);
          fprintf(log, "Resolution: %dx%d (render)  |  Device: Apple GPU\n", State::dispWIDTH, State::dispHEIGHT);
          fprintf(log, "Frames sampled: 50\n");
          fprintf(log, "═══════════════════════════════════════════════════════════════\n\n");

          // Calculate statistics
          const char* passNames[] = {"DistApprox", "GBuffer", "Indirect", "Accumulate", "Denoise", "Volumetric", "Exposure", "Composite", "TextOverlay", "MetalFX"};
          double avgs[10], mins[10], maxs[10], stds[10];
          for (int p = 0; p < 10; p++) {
            avgs[p] = mins[p] = maxs[p] = benchmark.data[0][p];
            for (int f = 1; f < 50; f++) {
              avgs[p] += benchmark.data[f][p];
              if (benchmark.data[f][p] < mins[p]) mins[p] = benchmark.data[f][p];
              if (benchmark.data[f][p] > maxs[p]) maxs[p] = benchmark.data[f][p];
            }
            avgs[p] /= 50.0;
            double variance = 0;
            for (int f = 0; f < 50; f++) {
              double diff = benchmark.data[f][p] - avgs[p];
              variance += diff * diff;
            }
            stds[p] = sqrt(variance / 50.0);
          }

          // Find bottleneck (pass with highest average)
          int bottleneckIdx = 2; // Indirect is typically heaviest
          for (int p = 0; p < 10; p++) {
            if (avgs[p] > avgs[bottleneckIdx]) bottleneckIdx = p;
          }
          double gpuTotal = avgs[0] + avgs[1] + avgs[2] + avgs[3] + avgs[4] + avgs[5] + avgs[6] + avgs[7] + avgs[8] + avgs[9];

          // GPU Pass Timings
          fprintf(log, "GPU Pass Timings (ms) — First 50 Frames Average:\n");
          for (int p = 0; p < 10; p++) {
            double pct = (avgs[p] / gpuTotal) * 100.0;
            fprintf(log, "  Pass %d  %-12s %6.2f ±%.2f    [%5.1f%%]%s\n",
                    p, passNames[p], avgs[p], stds[p], pct,
                    (p == bottleneckIdx) ? "  ◀ BOTTLENECK" : "");
          }
          fprintf(log, "  ──────────────────────────────────────────────────\n");
          fprintf(log, "  GPU Total              %6.2f\n\n", gpuTotal);

          // CPU Timings
          double cpuTextAvg = 0, cpuStreamAvg = 0, cpuDrawAvg = 0;
          for (int f = 0; f < 50; f++) {
            cpuTextAvg += benchmark.cpuTextPrep[f];
            cpuStreamAvg += benchmark.cpuStreaming[f];
            cpuDrawAvg += benchmark.cpuDrawTotal[f];
          }
          cpuTextAvg /= 50.0; cpuStreamAvg /= 50.0; cpuDrawAvg /= 50.0;
          fprintf(log, "CPU Timings (ms):\n");
          fprintf(log, "  Text Prep        %.2f\n", cpuTextAvg);
          fprintf(log, "  Streaming        %.2f\n", cpuStreamAvg);
          fprintf(log, "  Draw Total       %.2f\n\n", cpuDrawAvg);

          // Per-frame breakdown table
          fprintf(log, "Per-Frame Data:\n");
          fprintf(log, "  Frame  Approx  GBuff   Ind    Acc   Den   Fog  Expos  Cmp   Txt   MFX   Total  CPU\n");
          for (int f = 0; f < 50; f++) {
            fprintf(log, "  %04d  %5.2f  %5.2f %6.2f  %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %6.2f %5.2f\n",
                    f + 1,
                    benchmark.data[f][0], benchmark.data[f][1], benchmark.data[f][2],
                    benchmark.data[f][3], benchmark.data[f][4], benchmark.data[f][5],
                    benchmark.data[f][6], benchmark.data[f][7], benchmark.data[f][8],
                    benchmark.data[f][9], benchmark.data[f][10],
                    benchmark.cpuDrawTotal[f]);
          }
          fprintf(log, "\n═══════════════════════════════════════════════════════════════\n");
          fclose(log);
          NSLog(@"[Benchmark] Written benchmark.log with 50 frames of data");
        }
      }
    }

    dispatch_async(dispatch_get_main_queue(), ^{
      double w = State::dispWIDTH / 2.0;
      double h = State::dispHEIGHT / 2.0;
      double numPixels = w * h;

      double seconds = (msApprox < 0.001 ? 0.001 : msApprox) / 1000.0;
      double gigaRays = (numPixels / seconds) / 1e9;

      // Fixed: Now includes Expos and TextOverlay in total
      double totalMs = msApprox + msGBuffer + msIndirect + msAccum + msDenoise +
                       msFog + msExpos + msComp + msTextOverlay + msFX;

      NSString *title =
          [NSString stringWithFormat:
                        @"RVGRT | Approx: %.2fms (%.2f Grays/s) | GBuff: %.2f "
                        @"| Ind: %.2f | Acc: %.2f | Den: %.2f | Fog: %.2f | "
                        @"Expos: %.2f | Cmp: %.2f | Txt: %.2f | MFX: %.2f | Total: %.2fms | CPU: %.2fms",
                        msApprox, gigaRays, msGBuffer, msIndirect, msAccum,
                        msDenoise, msFog, msExpos, msComp, msTextOverlay, msFX, totalMs,
                        renderer->cpuDrawTotalMs];

      [self->_window setTitle:title];

      // Update fpsInfo for /fps command
      State::state.fpsInfo = "Approx: " + std::to_string(msApprox) + "ms | Total: " +
                             std::to_string(totalMs) + "ms";
    });
  }];

  [commandBuffer presentDrawable:drawable];
  [commandBuffer commit];
}

// This delegate method is called when the user resizes the window.
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
  MetalRenderer *renderer =
      static_cast<MetalRenderer *>(State::state.renderer.get());
  if (renderer) {
    int newScreenW = static_cast<int>(size.width);
    int newScreenH = static_cast<int>(size.height);
    int newDispW = newScreenW / 2;
    int newDispH = newScreenH / 2;
    if (newDispW < 1) newDispW = 1;
    if (newDispH < 1) newDispH = 1;
    renderer->OnResize(newDispW, newDispH, newScreenW, newScreenH);
  }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  return YES;
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
  // Cleanup will be handled by the unique_ptr destructors when State is
  // destroyed. Any manual cleanup would go here.
}

@end

// The standard C-style entry point for the application.
int MacOSMain(int argc, const char *argv[]) {
  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];
    AppDelegate *delegate = [[AppDelegate alloc] init];
    app.delegate = delegate;
    [app run];
  }
  return 0;
}
