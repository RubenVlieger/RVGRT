#ifdef D3D12


#pragma once
#include "util.hpp"
#include "Character.hpp"
#include "hitInfo.hpp"
#include "Framebuffer.cuh"
#include <bitset>
#include "Timer.hpp"
#include <atomic>

// D3D12 specific headers
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h> // For Microsoft::WRL::ComPtr

#include "Texturepack.cuh"

class StateRender;

class State {
public:
    // --- Application & Window ---
    HWND hwnd;
    StateRender* render;
    static constexpr int dispHEIGHT = 2400;
    static constexpr int dispWIDTH = 3840;

    // --- Input & Timing ---
    glm::vec2 mouseDelta;
    float deltaTime = 16; // Default to ~60 FPS
    std::atomic<long> deltaXMouse{0};
    std::atomic<long> deltaYMouse{0};
    std::bitset<256> keysPressed;
    FrameTimeAverager frameTimeAverager;

    bool IsKeyDown(char key);
    glm::vec2 getMouseDelta();

    // --- Legacy Bitmap (if still needed) ---
    char* bmp = nullptr;
    void setBitMap(char* _bmp) { bmp = _bmp; }

    // --- 3D World ---
    Character character = Character();

    // --- D3D12 Core Objects ---
    static constexpr UINT g_frameCount = 2; // Number of back buffers for the swap chain

    Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
    Microsoft::WRL::ComPtr<ID3D12Resource> renderTargets[g_frameCount];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[g_frameCount]; 
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap;
    
    // --- D3D12 Synchronization ---
    UINT frameIndex;
    UINT rtvDescriptorSize;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue;
    HANDLE fenceEvent;
    UINT64 fenceValues[g_frameCount]; // <--- ADD THIS LINE


    // --- Lifecycle ---
    void Create();
    
    // --- Singleton Instance ---
    static State state;

    // --- Constructor ---
    State();

private:
    // Private members can be added here if needed
};


#else


#pragma once
#include "util.hpp"
#include "Character.hpp"
#include "hitInfo.hpp"
#include "Framebuffer.cuh"
#include <bitset>
#include "Timer.hpp"
#include "d3d11.h"
#include <atomic>

#include "Texturepack.cuh"

class StateRender;

class State {
public:
    HWND hwnd;
    StateRender* render;

    //Texturepack texturepack(texturepack_png, texturepack_png_len);

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
    static constexpr int dispHEIGHT = 1200;
    static constexpr int dispWIDTH = 1920;

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
#endif