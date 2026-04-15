#pragma once

#include "util.hpp"
#include "Character.hpp"
#include "hitInfo.hpp"
#include "Timer.hpp"
#include "console/GameConsole.hpp"
#include "BlockInteraction.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

    // --- Console & Game State ---
    GameConsole console;

    // Home position for /sethome and /home
    glm::vec3 homePosition = glm::vec3(508.0f, 156.0f, 408.0f);

    // Fly mode state (not yet functional)
    bool flyMode = false;

    // Noclip mode: free flight with no terrain collision (default: true)
    // When set to false, full terrain collision is enabled with gravity and jumping
    bool noclipMode = true;

    // Sun direction override (0,0,0 = use default)
    glm::vec3 sunDirectionOverride = glm::vec3(0.0f, 0.0f, 0.0f);

    // Last-frame FPS string for /fps command
    std::string fpsInfo = "0.0 ms";

    // --- Block Interaction (Phase 2) ---
    uint8_t selectedMaterialID = 2; // MAT_GRASS
    std::vector<BlockEdit> localBlockEdits;

    // Thread-safe queue for remote block edits (Phase 4 network)
    std::mutex blockEditsMutex;
    std::vector<BlockEdit> pendingRemoteEdits;
    bool blockResetRequested = false;

    // --- Singleton Instance ---
    static State state;

    // --- Constructor and Destructor ---
    State();
    ~State();
};