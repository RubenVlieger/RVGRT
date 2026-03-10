#pragma once

#include "renderer/ShaderTypes.h"
#include "Character.hpp"
#include <vector>
#include <chrono>

// Forward declaration of State class
class State;

/**
 * FrameDataManager - Unified data preparation for rendering
 * 
 * This class extracts the duplicated data preparation logic from both
 * MetalRenderer and CudaRenderer. It handles:
 * - Camera data preparation (position, direction, matrices)
 * - Frame data preparation (time, sun direction, world origin)
 * - Character data preparation (player + NPCs)
 * - Timing management (delta time calculation)
 */
class FrameDataManager {
public:
    FrameDataManager();
    
    /**
     * Prepare camera data from character state
     * @param character The player character
     * @return CameraData structure ready for GPU upload
     */
    CameraData PrepareCameraData(const Character& character);
    
    /**
     * Prepare frame-level data
     * @param frameCount Current frame count (for time calculation)
     * @param worldOrigin World origin for toroidal wrapping
     * @return FrameData structure ready for GPU upload
     */
    FrameData PrepareFrameData(unsigned int frameCount, simd_int3 worldOrigin);
    
    /**
     * Prepare character data for GPU (player + NPCs)
     * @param player The player character
     * @param npcs Vector of NPC characters
     * @return CharacterGPUData structure ready for GPU upload
     */
    CharacterGPUData PrepareCharacterData(const Character& player, 
                                          const std::vector<Character>& npcs);
    
    /**
     * Get the current delta time (time since last frame)
     * @return Delta time in seconds
     */
    float GetDeltaTime() const { return _deltaTime; }
    
    /**
     * Get the current time (modulo 3600 for shader precision)
     * @return Time in seconds
     */
    float GetCurrentTime() const { return _currentTime; }
    
    /**
     * Reset the timing state (call on scene change/teleport)
     */
    void ResetTiming();

private:
    // Timing state
    double _lastTime;
    float _deltaTime;
    float _currentTime;
    bool _firstFrame;
    
    // Sun direction (constant for now, could be made dynamic)
    static constexpr simd_float3 SUN_DIRECTION = {10.0f, 5.0f, -4.0f};
    
    // Helper to append a single character's data
    void AppendCharacterData(CharacterGPUData& data, const Character& character, int& activeCount);
};
