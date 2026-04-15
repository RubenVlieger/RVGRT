#include "Character.hpp"
#include "Camera.hpp"
#include "State.hpp"
#include "platform/Platform.hpp"
#include "VoxelQuery.hpp"
#include "util.hpp"
#include <cmath>
#include <numbers>

#include <glm/gtc/matrix_transform.hpp>

float halton(int index, int base) {
  float f = 1.0f;
  float r = 0.0f;
  while (index > 0) {
    f = f / (float)base;
    r = r + f * (float)(index % base);
    index = index / base;
  }
  return r * 0.5f;
}

Character::Character()
    : walkPhase(0.0f), walkSwingAmount(0.0f), viewMatrix(1.0f),
      projectionMatrix(1.0f), viewProjectionMatrix(1.0f),
      unjitteredViewProjectionMatrix(1.0f),
      prevUnjitteredViewProjectionMatrix(1.0f),
      inverseViewProjectionMatrix(1.0f), prevViewProjectionMatrix(1.0f),
      nearPlane(0.1f), farPlane(1000.0f), FOV(70.0f), jitterX(0.0f),
      jitterY(0.0f), lockMouse(false),
      position(508.0f, 156.0f, 408.0f), // Start in the middle of the world
      velocity(0.0f), direction(0.0f, 0.0f, -1.0f),
      yaw(std::numbers::pi_v<float> * -0.5f), pitch(-std::numbers::pi_v<float>),
      speed(0.05f), speedDropoff(0.92f), jumpSpeed(2.0f), sensitivity(0.00003f),
      gravityAmount(0.0f),
      onGround(false), playerHeight(1.62f), playerRadius(0.3f) {
  lastRenderedViewProjectionMatrix = glm::mat4(1.0f);
}

void Character::UpdateTestNPC(float time, float deltaTime) {
  // Move back and forth along the Z axis based on a sine wave.
  // We place it slightly in front of the typical spawn point (408.0f)
  float baseZ = 405.0f;
  float newZ = baseZ + sin(time * 2.0f) * 3.0f;

  // Fake velocity to drive the animation
  velocity.x = 0.0f;
  velocity.y = 0.0f;
  velocity.z = (newZ - position.z) / fmax(deltaTime, 0.001f);

  position.x = 508.0f;
  position.y = 156.0f;
  position.z = newZ;

  // Face the direction of movement
  if (velocity.z > 0.0f) {
    direction = glm::dvec3(0.0, 0.0, 1.0);
  } else {
    direction = glm::dvec3(0.0, 0.0, -1.0);
  }

  vec2 horizontalVelocity = vec2(velocity.x, velocity.z);
  float currentSpeed = length(horizontalVelocity);

  float targetSwing = glm::clamp(currentSpeed * 2.0f, 0.0f, 1.0f);
  walkSwingAmount = glm::mix(walkSwingAmount, targetSwing, deltaTime * 10.0f);

  walkPhase += currentSpeed * deltaTime * 25.0f;
  walkPhase = fmod(walkPhase, std::numbers::pi_v<float> * 2.0f);

  UpdateTransformations();
}

glm::dvec3 calcDirfromSphere(double pitch, double yaw) {
  const float pih = std::numbers::pi_v<float> * 0.5f;
  glm::vec4 sins =
      glm::vec4(sin(yaw), sin(yaw + pih), sin(pitch), sin(pitch + pih));
  return normalize(
      glm::vec3(-sins[0] * -sins[3], -sins[2], -sins[1] * sins[3]));
}

void Character::Update(unsigned int frameCount) {
  Platform *platform = State::state.platform.get();
  if (!platform)
    return; // Guard against calls before platform is initialized

  prevViewProjectionMatrix = viewProjectionMatrix;
  prevUnjitteredViewProjectionMatrix = unjitteredViewProjectionMatrix;

  float dt = platform->deltaTime / 1000.0f;

  if (!lockMouse) {
    float deltayaw = platform->deltaXMouse.exchange(0) * sensitivity *
                     platform->deltaTime * FOV;
    float deltapitch = platform->deltaYMouse.exchange(0) * sensitivity *
                       platform->deltaTime * FOV;

    if (platform->IsKeyDown(0x38)) { // Shift modifier
      deltayaw *= 0.4f;
      deltapitch *= 0.4f;
    }

    yaw += deltayaw;
    pitch += deltapitch;

    yaw = fmod(yaw, std::numbers::pi * 2.0f);
    pitch = clamp(pitch, -4.5f, -1.65f);
    direction = calcDirfromSphere(pitch, yaw);
  }

  if (State::state.noclipMode) {
    // ─────────────────────────────────────────────────────────────────
    // NOCLIP MODE: Free flight, no terrain collision
    // ─────────────────────────────────────────────────────────────────
    vec3 inputs = vec3((platform->IsKeyDown('D') ? 1.0f : 0.0f) +
                           (platform->IsKeyDown('A') ? -1.0f : 0.0f),
                       (platform->IsKeyDown(' ') ? 1.0f : 0.0f) +
                           (platform->IsKeyDown('Z') ? -1.0f : 0.0f),
                       (platform->IsKeyDown('W') ? 1.0f : 0.0f) +
                           (platform->IsKeyDown('S') ? -1.0f : 0.0f)) *
                  speed;

    if (platform->IsKeyDown(0x38)) { // Shift modifier
      inputs *= 0.3f;
    }

    velocity += inputs.x * glm::cross((vec3)direction, vec3(0.0f, 1.0f, 0.0f)) +
                inputs.z * (vec3)direction;
    velocity *= speedDropoff;

    vec3 addVector = velocity * platform->deltaTime;
    position = glm::mix(position, position + addVector, 0.5f);

  } else {
    // ─────────────────────────────────────────────────────────────────
    // CLIPPED MODE: Full terrain collision with gravity and jumping
    // ─────────────────────────────────────────────────────────────────
    constexpr float walkSpeed = 4.3f;  // blocks per second
    constexpr float sprintSpeed = 6.5f;
    constexpr float terminalVelocity = -50.0f;
    constexpr float stepHeight = 0.6f; // max step up height

    // Horizontal movement — direct velocity control, not accumulation
    float hSpeed = platform->IsKeyDown(0x38) ? sprintSpeed : walkSpeed;

    vec3 moveDir(0.0f, 0.0f, 0.0f);
    if (platform->IsKeyDown('W')) moveDir.z -= 1.0f;
    if (platform->IsKeyDown('S')) moveDir.z += 1.0f;
    if (platform->IsKeyDown('A')) moveDir.x -= 1.0f;
    if (platform->IsKeyDown('D')) moveDir.x += 1.0f;

    float moveLen = sqrt(moveDir.x * moveDir.x + moveDir.z * moveDir.z);
    if (moveLen > 0.01f) {
      moveDir.x /= moveLen;
      moveDir.z /= moveLen;
    }

    // Convert moveDir from camera-relative to world XZ
    vec3 dirRight = normalize(cross((vec3)direction, vec3(0.f, 1.f, 0.f)));
    velocity.x = moveDir.x * dirRight.x * hSpeed + moveDir.z * (-direction.x) * hSpeed;
    velocity.z = moveDir.x * dirRight.z * hSpeed + moveDir.z * (-direction.z) * hSpeed;

    // Jump impulse
    if (platform->IsKeyDown(' ') && onGround) {
      velocity.y = jumpSpeed;
      onGround = false;
    }

    // Gravity
    velocity.y += gravityAmount * dt;

    // Terminal velocity clamp
    velocity.y = glm::min(velocity.y, terminalVelocity);

    // Propose new position
    glm::vec3 proposedPos = position + velocity * dt;

    // ── AABB Collision Resolution ──────────────────────────────────
    //
    // Player AABB: feet at position.y - playerHeight, head at position.y + 0.18f
    //              half-width = playerRadius on X and Z
    //
    // Algorithm: resolve each axis independently (Y → X → Z) to allow
    // wall-sliding and prevent corner-sticking.

    float feetY = position.y - playerHeight;
    float headRoom = 0.18f; // eye-to-head clearance

    // Helper: check if a voxel at (mx, my, mz) is inside the player's AABB
    // at given Y (for Y checks) or at proposed Y (for XZ checks)
    auto overlapsAABB = [&](int mx, int my, int mz,
                             float aabbMinY, float aabbMaxY,
                             float aabbMinX, float aabbMaxX,
                             float aabbMinZ, float aabbMaxZ) -> bool {
      return (mx + 1 > aabbMinX && mx < aabbMaxX &&
              my + 1 > aabbMinY && my < aabbMaxY &&
              mz + 1 > aabbMinZ && mz < aabbMaxZ);
    };

    // ── Y-Axis Collision (ground + ceiling) ────────────────────────
    {
      float resolvedFeetY = proposedPos.y - playerHeight;
      bool hitGround = false;
      bool hitCeiling = false;

      // Check voxels the player would occupy at proposed Y
      int minIX = int(floor(proposedPos.x - playerRadius));
      int maxIX = int(floor(proposedPos.x + playerRadius));
      int minIZ = int(floor(proposedPos.z - playerRadius));
      int maxIZ = int(floor(proposedPos.z + playerRadius));

      // Ground check: look for solid voxels directly under feet
      int groundY = int(floor(resolvedFeetY));
      for (int ix = minIX; ix <= maxIX; ++ix) {
        for (int iz = minIZ; iz <= maxIZ; ++iz) {
          if (IsVoxelSolid(ix, groundY, iz)) {
            hitGround = true;
            break;
          }
        }
        if (hitGround) break;
      }

      if (hitGround) {
        // Snap feet to top of solid voxel
        position.y = float(groundY + 1) + playerHeight;
        velocity.y = 0.0f;
        onGround = true;
      } else {
        // Ceiling check: look for solid voxels above head at proposed Y
        position.y = proposedPos.y;
        onGround = false;

        int headY = int(floor(proposedPos.y + headRoom));
        for (int ix = minIX; ix <= maxIX; ++ix) {
          for (int iz = minIZ; iz <= maxIZ; ++iz) {
            if (IsVoxelSolid(ix, headY, iz)) {
              hitCeiling = true;
              break;
            }
          }
          if (hitCeiling) break;
        }
        if (hitCeiling) {
          int ceilingY = int(floor(proposedPos.y + headRoom));
          position.y = float(ceilingY) - headRoom;
          velocity.y = 0.0f;
        }
      }
    }

    // ── X-Axis Collision ─────────────────────────────────────────────
    {
      float resolvedY = position.y;
      float aabbMinY = resolvedY - playerHeight;
      float aabbMaxY = resolvedY + headRoom;
      float aabbMinZ = position.z - playerRadius;
      float aabbMaxZ = position.z + playerRadius;

      int minIX = int(floor(proposedPos.x - playerRadius));
      int maxIX = int(floor(proposedPos.x + playerRadius));
      int minIY = int(floor(aabbMinY));
      int maxIY = int(floor(aabbMaxY));
      int minIZ = int(floor(aabbMinZ));
      int maxIZ = int(floor(aabbMaxZ));

      bool hitX = false;
      int hitVoxelX = 0;

      if (proposedPos.x > position.x) {
        // Moving positive X — check voxels on the +X face
        int checkX = int(floor(proposedPos.x + playerRadius));
        for (int iy = minIY; iy <= maxIY; ++iy) {
          for (int iz = minIZ; iz <= maxIZ; ++iz) {
            if (IsVoxelSolid(checkX, iy, iz)) {
              hitX = true;
              hitVoxelX = checkX;
              break;
            }
          }
          if (hitX) break;
        }
        if (hitX) {
          position.x = float(hitVoxelX) - playerRadius;
          velocity.x = 0.0f;
        } else {
          position.x = proposedPos.x;
        }
      } else if (proposedPos.x < position.x) {
        // Moving negative X — check voxels on the -X face
        int checkX = int(floor(proposedPos.x - playerRadius));
        for (int iy = minIY; iy <= maxIY; ++iy) {
          for (int iz = minIZ; iz <= maxIZ; ++iz) {
            if (IsVoxelSolid(checkX, iy, iz)) {
              hitX = true;
              hitVoxelX = checkX;
              break;
            }
          }
          if (hitX) break;
        }
        if (hitX) {
          position.x = float(hitVoxelX + 1) + playerRadius;
          velocity.x = 0.0f;
        } else {
          position.x = proposedPos.x;
        }
      }
    }

    // ── Z-Axis Collision ─────────────────────────────────────────────
    {
      float aabbMinY = position.y - playerHeight;
      float aabbMaxY = position.y + headRoom;
      float aabbMinX = position.x - playerRadius;
      float aabbMaxX = position.x + playerRadius;

      int minIZ = int(floor(proposedPos.z - playerRadius));
      int maxIZ = int(floor(proposedPos.z + playerRadius));
      int minIY = int(floor(aabbMinY));
      int maxIY = int(floor(aabbMaxY));
      int minIX = int(floor(aabbMinX));
      int maxIX = int(floor(aabbMaxX));

      bool hitZ = false;
      int hitVoxelZ = 0;

      if (proposedPos.z > position.z) {
        // Moving positive Z — check voxels on the +Z face
        int checkZ = int(floor(proposedPos.z + playerRadius));
        for (int iy = minIY; iy <= maxIY; ++iy) {
          for (int ix = minIX; ix <= maxIX; ++ix) {
            if (IsVoxelSolid(ix, iy, checkZ)) {
              hitZ = true;
              hitVoxelZ = checkZ;
              break;
            }
          }
          if (hitZ) break;
        }
        if (hitZ) {
          position.z = float(hitVoxelZ) - playerRadius;
          velocity.z = 0.0f;
        } else {
          position.z = proposedPos.z;
        }
      } else if (proposedPos.z < position.z) {
        // Moving negative Z — check voxels on the -Z face
        int checkZ = int(floor(proposedPos.z - playerRadius));
        for (int iy = minIY; iy <= maxIY; ++iy) {
          for (int ix = minIX; ix <= maxIX; ++ix) {
            if (IsVoxelSolid(ix, iy, checkZ)) {
              hitZ = true;
              hitVoxelZ = checkZ;
              break;
            }
          }
          if (hitZ) break;
        }
        if (hitZ) {
          position.z = float(hitVoxelZ + 1) + playerRadius;
          velocity.z = 0.0f;
        } else {
          position.z = proposedPos.z;
        }
      }
    }

    // Re-check onGround: if Y resolution placed player on ground
    {
      int groundY = int(floor(position.y - playerHeight));
      int minIX = int(floor(position.x - playerRadius));
      int maxIX = int(floor(position.x + playerRadius));
      int minIZ = int(floor(position.z - playerRadius));
      int maxIZ = int(floor(position.z + playerRadius));
      for (int ix = minIX; ix <= maxIX; ++ix) {
        for (int iz = minIZ; iz <= maxIZ; ++iz) {
          if (IsVoxelSolid(ix, groundY, iz)) {
            onGround = true;
            break;
          }
        }
        if (onGround) break;
      }
    }
  }

  // ── Camera & View Matrices (shared by both modes) ──────────────────
  vec3 dirright = normalize(cross((vec3)direction, vec3(0.f, 1.f, 0.f)));
  vec3 dirup = normalize(cross((vec3)direction, dirright));

  viewMatrix = glm::lookAt(position, position + (vec3)direction,
                           glm::vec3(0.0f, 1.0f, 0.0f));

  projectionMatrix = glm::perspective(
      glm::radians(FOV), (float)State::dispWIDTH / (float)State::dispHEIGHT,
      nearPlane, farPlane);
  unjitteredViewProjectionMatrix = projectionMatrix * viewMatrix;

  int jitterIndex = (frameCount % 16) + 1;
  jitterX = halton(jitterIndex, 2) - 0.5f;
  jitterY = halton(jitterIndex, 3) - 0.5f;

  float clipSpaceJitterX = jitterX / (0.5f * State::dispWIDTH);
  float clipSpaceJitterY = jitterY / (0.5f * State::dispHEIGHT);
  projectionMatrix[2][0] += clipSpaceJitterX;
  projectionMatrix[2][1] += clipSpaceJitterY;

  viewProjectionMatrix = projectionMatrix * viewMatrix;
  inverseViewProjectionMatrix = glm::inverse(unjitteredViewProjectionMatrix);

  camera.pos = position;
  camera.forward = vec3(direction);
  camera.right = dirright;
  camera.up = dirup;

  platform->deltaXMouse.store(0);
  platform->deltaYMouse.store(0);

  // ── Walk Animation Phase ────────────────────────────────────────────
  vec2 horizontalVelocity = vec2(velocity.x, velocity.z);
  float currentSpeed = length(horizontalVelocity);

  float targetSwing = glm::clamp(currentSpeed * 2.0f, 0.0f, 1.0f);
  walkSwingAmount =
      glm::mix(walkSwingAmount, targetSwing, platform->deltaTime * 10.0f);

  walkPhase += currentSpeed * platform->deltaTime * 25.0f;
  walkPhase = fmod(walkPhase, std::numbers::pi_v<float> * 2.0f);

  UpdateTransformations();
}

bool Character::IsKeyDown(char keycode) {
  if (State::state.platform) {
    return State::state.platform->IsKeyDown(keycode);
  }
  return false;
}

static glm::mat4 BuildPartMatrix(glm::vec3 basePosition, float baseYaw,
                                 glm::vec3 hingeOffset,
                                 glm::vec3 partCenterOffset, glm::vec3 partSize,
                                 float pitchLocal) {
  glm::mat4 m(1.0f);

  // 1. Position in world
  m = glm::translate(m, basePosition);

  // 2. Base Character Yaw (orient whole character body towards camera
  // direction) Positive right-handed rotation around Y faces -Z towards the
  // calculated bodyYaw direction
  m = glm::rotate(m, baseYaw, glm::vec3(0.0f, 1.0f, 0.0f));

  // 3. Move relative to the hinge of this part
  m = glm::translate(m, hingeOffset);

  // 4. Apply hinge rotation (walk animation or head pitch)
  if (pitchLocal != 0.0f) {
    m = glm::rotate(m, pitchLocal, glm::vec3(1.0f, 0.0f, 0.0f));
  }

  // 5. Offset internal center relative to the hinge
  m = glm::translate(m, partCenterOffset);

  // 6. Scale out to the required AABB size
  m = glm::scale(m, partSize);

  return m;
}

void Character::UpdateTransformations() {
  // Determine horizontal facing direction for the whole body
  // -Z is forward in this coordinate system.
  float bodyYaw = std::atan2(direction.x, -direction.z);

  // Calculate head pitch decoupled from the global pitch state var
  float headPitch = (float)std::asin(glm::clamp((float)direction.y, -1.0f, 1.0f));

  // 'position' represents the camera/eye level. In our 2.0 tall character, eyes
  // are at ~1.62.
  glm::vec3 feetPosition = position - glm::vec3(0.0f, 1.62f, 0.0f);

  // Max swing rotation ~45 degrees (0.8 radians) scaled by our current
  // momentum.
  float swingAngle = sin(walkPhase) * 0.8f * walkSwingAmount;

  // Overall Character Bounding Box (for coarse culling)
  glm::mat4 bboxMat = glm::translate(
      glm::mat4(1.0f), feetPosition + glm::vec3(0.0f, 1.0f, 0.0f));
  bboxMat = glm::rotate(bboxMat, bodyYaw, glm::vec3(0.0f, 1.0f, 0.0f));
  bboxMat = glm::scale(bboxMat, glm::vec3(2.4f, 2.2f, 2.4f));
  boundingBox.modelMatrix = bboxMat;
  boundingBox.inverseModelMatrix = glm::inverse(bboxMat);

  // Head (Hinges at the bottom of the head, y=1.5)
  head.modelMatrix = BuildPartMatrix(
      feetPosition, bodyYaw, glm::vec3(0.0f, 1.5f, 0.0f),
      glm::vec3(0.0f, 0.25f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f), headPitch);
  head.inverseModelMatrix = glm::inverse(head.modelMatrix);

  // Trunk (No independent swing, just body yaw)
  trunk.modelMatrix = BuildPartMatrix(
      feetPosition, bodyYaw, glm::vec3(0.0f, 1.5f, 0.0f),
      glm::vec3(0.0f, -0.375f, 0.0f), glm::vec3(0.5f, 0.75f, 0.25f), 0.0f);
  trunk.inverseModelMatrix = glm::inverse(trunk.modelMatrix);

  // Left Arm (Opposite swing to left leg logic)
  leftArm.modelMatrix =
      BuildPartMatrix(feetPosition, bodyYaw, glm::vec3(-0.375f, 1.5f, 0.0f),
                      glm::vec3(0.0f, -0.375f, 0.0f),
                      glm::vec3(0.25f, 0.75f, 0.25f), -swingAngle);
  leftArm.inverseModelMatrix = glm::inverse(leftArm.modelMatrix);

  // Right Arm
  rightArm.modelMatrix =
      BuildPartMatrix(feetPosition, bodyYaw, glm::vec3(0.375f, 1.5f, 0.0f),
                      glm::vec3(0.0f, -0.375f, 0.0f),
                      glm::vec3(0.25f, 0.75f, 0.25f), swingAngle);
  rightArm.inverseModelMatrix = glm::inverse(rightArm.modelMatrix);

  // Left Leg
  leftLeg.modelMatrix =
      BuildPartMatrix(feetPosition, bodyYaw, glm::vec3(-0.125f, 0.75f, 0.0f),
                      glm::vec3(0.0f, -0.375f, 0.0f),
                      glm::vec3(0.25f, 0.75f, 0.25f), swingAngle);
  leftLeg.inverseModelMatrix = glm::inverse(leftLeg.modelMatrix);

  // Right Leg
  rightLeg.modelMatrix =
      BuildPartMatrix(feetPosition, bodyYaw, glm::vec3(0.125f, 0.75f, 0.0f),
                      glm::vec3(0.0f, -0.375f, 0.0f),
                      glm::vec3(0.25f, 0.75f, 0.25f), -swingAngle);
  rightLeg.inverseModelMatrix = glm::inverse(rightLeg.modelMatrix);
}