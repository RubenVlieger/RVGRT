#pragma once
#include "util.hpp"
#include "Character.hpp"
#include "hitInfo.hpp"
#include "Framebuffer.cuh"
#include <bitset>


class StateRender;

class State {
public:
    StateRender* render;
    // Input & mouse
    glm::vec2 mouseDelta;
    float deltaTime = 16;
    float deltaXMouse;
    float deltaYMouse;
    std::bitset<256> keysPressed;

    bool IsKeyDown(char key);
    glm::vec2 getMouseDelta();

    // Bitmap / framebuffer
    char* bmp = nullptr;
    void setBitMap(char* _bmp) { bmp = _bmp; }

    // 3D world
    Character character = Character();
    static constexpr int dispHEIGHT = 480*2;
    static constexpr int dispWIDTH  = 640*2;

    void Create();

    static State state;

    // Constructor
    State();

private:
};
