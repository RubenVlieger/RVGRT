#pragma once
#include "Timer.hpp"
#include "util.hpp"
#include <atomic>
#include <bitset>

class Platform {
public:
  virtual ~Platform() = default;

  virtual void Create() = 0;
  virtual bool IsKeyDown(char keycode) = 0;

  // Platform-agnostic way to get window and view handles
  virtual void *GetWindowHandle() = 0; // HWND on Windows, NSWindow* on macOS
  virtual void *GetViewHandle() = 0;   // HWND on Windows, MTKView* on macOS

  // --- Input & Timing ---
  float deltaTime = 16;
  std::atomic<long> deltaXMouse{0};
  std::atomic<long> deltaYMouse{0};
  std::bitset<256> keysPressed;
  FrameTimeAverager frameTimeAverager;
};