#pragma once
#include "util.hpp"
#include "Character.hpp"
#include "hitInfo.hpp"
#include "Framebuffer.cuh"
#include <bitset>
#include "Timer.hpp"
#include "d3d11.h"
#include <atomic>

class StateRender;

class State {
public:
    HWND hwnd;
    StateRender* render;
    // Input & mouse
    glm::vec2 mouseDelta;
    float deltaTime = 16;
    std::atomic<long> deltaXMouse{0};
    std::atomic<long> deltaYMouse{0};
    std::bitset<256> keysPressed;
    FrameTimeAverager frameTimeAverager;

    bool IsKeyDown(char key);
    glm::vec2 getMouseDelta();

    // Bitmap / framebuffer
    char* bmp = nullptr;
    void setBitMap(char* _bmp) { bmp = _bmp; }

    // 3D world
    Character character = Character();
    static constexpr int dispHEIGHT = 1200*2;
    static constexpr int dispWIDTH = 1920*2;

    void Create();

    static State state;

    // Constructor
    State();

    IDXGISwapChain* swapChain;
    ID3D11Device* d3dDevice;
    ID3D11DeviceContext* d3dContext;
    ID3D11Texture2D* backBufferTexture;


private:

};
