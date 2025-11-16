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
    static constexpr int dispHEIGHT = 1920 * 0.6f; //800; //3072 x 1920
    static constexpr int dispWIDTH = 3072 * 0.6f; //1280;
    static constexpr int screenHEIGHT = 1920 * 0.6f;
    static constexpr int screenWIDTH = 3072 * 0.6f;

    // --- 3D World ---
    Character character = Character();

    // --- Singleton Instance ---
    static State state;

    // --- Constructor and Destructor ---
    State();
    ~State(); 

};