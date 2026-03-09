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
class NetworkClient;

class State {
public:
    // --- Platform and Renderer Abstractions ---
    std::unique_ptr<Platform> platform;
    std::unique_ptr<GraphicsDevice> graphicsDevice;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<NetworkClient> networkClient;

    // --- Resolution Control ---
    static int dispHEIGHT;
    static int dispWIDTH;
    static int screenHEIGHT;
    static int screenWIDTH;

    // --- 3D World ---
    Character character = Character();
    std::vector<Character> otherCharacters;

    // --- Singleton Instance ---
    static State state;

    // --- Constructor and Destructor ---
    State();
    ~State(); 

};