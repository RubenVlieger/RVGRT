#include "renderer/FrameDataManager.hpp"
#include "State.hpp"
#include "platform/Platform.hpp"
#include "cumath.h"
#include <cmath>
#include <cstring>

FrameDataManager::FrameDataManager()
    : _lastTime(0.0), _deltaTime(0.001f), _currentTime(0.0f),
      _firstFrame(true) {}

CameraData FrameDataManager::PrepareCameraData(const Character &character) {
  CameraData camData;

  // Position
  camData.position = simd_make_float3(static_cast<float>(character.position.x),
                                      static_cast<float>(character.position.y),
                                      static_cast<float>(character.position.z));

  // Forward direction
  camData.forward = simd_make_float3(static_cast<float>(character.direction.x),
                                     static_cast<float>(character.direction.y),
                                     static_cast<float>(character.direction.z));

  // Calculate right/up vectors based on FOV and aspect ratio
  float tanHalfFov = tanf(glm::radians(character.FOV) * 0.5f);
  float aspect = static_cast<float>(State::dispWIDTH) /
                 static_cast<float>(State::dispHEIGHT);

  glm::vec3 sRight = character.camera.right * tanHalfFov * aspect;
  glm::vec3 sUp = character.camera.up * tanHalfFov;

  camData.right = simd_make_float3(sRight.x, sRight.y, sRight.z);
  camData.up = simd_make_float3(sUp.x, sUp.y, sUp.z);

  // Jitter for TAA
  camData.jitter = simd_make_float2(character.jitterX, character.jitterY);

  // View projection matrices
  memcpy(&camData.unjitteredViewProjection,
         &character.unjitteredViewProjectionMatrix, sizeof(simd_float4x4));

  memcpy(&camData.prevUnjitteredViewProjection,
         &character.prevUnjitteredViewProjectionMatrix, sizeof(simd_float4x4));

  return camData;
}

FrameData FrameDataManager::PrepareFrameData(unsigned int frameCount,
                                             simd_int3 worldOrigin) {
  FrameData frameData;

  // Sun direction (normalized)
  frameData.sunDirection = simd_normalize(SUN_DIRECTION);

  // Time calculation (using frame count for deterministic behavior)
  // Frame count / 60.0 gives approximate seconds at 60fps
  _currentTime = fmodf(static_cast<float>(frameCount) / 60.0f, 3600.0f);
  frameData.time = _currentTime;

  // Calculate delta time
  if (_firstFrame) {
    _deltaTime = 1.0f / 60.0f; // Assume 60fps for first frame
    _firstFrame = false;
  } else {
    // Access elapsed time from Platform if available
    if (State::state.platform) {
      _deltaTime =
          static_cast<float>(State::state.platform->deltaTime / 1000.0);
    }

    // Safety bounds
    _deltaTime = max(_deltaTime, 0.001f); // minimum 1ms
    _deltaTime =
        min(_deltaTime, 0.1f); // cap at 10fps to avoid massive exposure jumps
  }
  frameData.deltaTime = _deltaTime;

  // World origin for toroidal wrapping
  frameData.worldOrigin = worldOrigin;

  return frameData;
}

void FrameDataManager::AppendCharacterData(CharacterGPUData &data,
                                           const Character &character,
                                           int &activeCount) {
  if (activeCount >= MAX_CHARACTERS) {
    return;
  }

  int idx = activeCount;

  data.characterCenters[idx] =
      simd_make_float4(static_cast<float>(character.position.x),
                       static_cast<float>(character.position.y),
                       static_cast<float>(character.position.z), 0.0f);

  // Bounding box inverse matrix
  memcpy(&data.invBoundingBoxes[idx], &character.boundingBox.inverseModelMatrix,
         sizeof(simd_float4x4));

  // Body parts (6 per character: head, trunk, left arm, right arm, left leg,
  // right leg)
  memcpy(&data.invBodyParts[idx * 6 + 0], &character.head.inverseModelMatrix,
         sizeof(simd_float4x4));
  memcpy(&data.invBodyParts[idx * 6 + 1], &character.trunk.inverseModelMatrix,
         sizeof(simd_float4x4));
  memcpy(&data.invBodyParts[idx * 6 + 2], &character.leftArm.inverseModelMatrix,
         sizeof(simd_float4x4));
  memcpy(&data.invBodyParts[idx * 6 + 3],
         &character.rightArm.inverseModelMatrix, sizeof(simd_float4x4));
  memcpy(&data.invBodyParts[idx * 6 + 4], &character.leftLeg.inverseModelMatrix,
         sizeof(simd_float4x4));
  memcpy(&data.invBodyParts[idx * 6 + 5],
         &character.rightLeg.inverseModelMatrix, sizeof(simd_float4x4));

  activeCount++;
}

CharacterGPUData
FrameDataManager::PrepareCharacterData(const Character &player,
                                       const std::vector<Character> &npcs) {
  CharacterGPUData data;
  memset(&data, 0, sizeof(CharacterGPUData));

  int activeCount = 0;

  // Add player character
  AppendCharacterData(data, player, activeCount);

  // Add NPCs
  for (const auto &npc : npcs) {
    AppendCharacterData(data, npc, activeCount);
    if (activeCount >= MAX_CHARACTERS) {
      break;
    }
  }

  data.numCharacters = activeCount;

  return data;
}

void FrameDataManager::ResetTiming() {
  _firstFrame = true;
  _lastTime = 0.0;
  _deltaTime = 0.001f;
  _currentTime = 0.0f;
}
