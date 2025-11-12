#pragma once

#include "util.hpp"
#include "Character.hpp"
#include "hitInfo.hpp"
#include "Timer.hpp"
#include <memory>

// Forward-declare the abstract interfaces
class Platform;
class GraphicsDevice;
class Renderer;

class State {
public:
    // --- Platform and Renderer Abstractions ---
    std::unique_ptr<Platform> platform;
    std::unique_ptr<GraphicsDevice> graphicsDevice;
    std::unique_ptr<Renderer> renderer;

    // --- Resolution Control ---
    static constexpr int dispHEIGHT = 800;
    static constexpr int dispWIDTH = 1280;
    static constexpr int screenHEIGHT = 2400;
    static constexpr int screenWIDTH = 3840;

    // --- 3D World ---
    Character character = Character();

    // --- Singleton Instance ---
    static State state;

    // --- Constructor and Destructor ---
    State();
    ~State(); // <-- DECLARE the destructor here

};