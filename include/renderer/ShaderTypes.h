#ifndef ShaderTypes_h
#define ShaderTypes_h

// Include shared system configuration first
#include "SystemConfig.h"

// The simd library provides C++ types like `simd_float3` that are
// directly memory-compatible with Metal's `float3`.
#if defined(__METAL_VERSION__)
// Metal shader path - use Metal's native types
#include <metal_stdlib>
using namespace metal;
// Metal shaders use float4x4 directly, not simd_float4x4
using simd_float4x4 = float4x4;
using simd_float3 = float3;
using simd_float4 = float4;
using simd_float2 = float2;
using simd_int3 = int3;
#elif defined(__APPLE__)
// macOS C++ path - use system simd library
#include <simd/simd.h>
#else
// Windows/CUDA path - use cumath types
#include "../cumath.h"
using simd_float2 = float2;
using simd_float3 = float3;
using simd_float4 = float4;
using simd_int3 = int3;
using simd_float4x4 = mat4;

// Utility functions that Metal provides natively
inline float3 simd_normalize(float3 v) { return normalize(v); }
inline float3 simd_make_float3(float x, float y, float z) { return make_float3(x, y, z); }
inline float2 simd_make_float2(float x, float y) { return make_float2(x, y); }
#endif

// This struct is now defined in a plain C++ header.
// Both your .metal file and your .mm file will include this.
struct CameraData {
  simd_float3 position;
  simd_float3 forward;
  simd_float3 right;
  simd_float3 up;

  simd_float4x4 unjitteredViewProjection;
  simd_float4x4 prevUnjitteredViewProjection;

  simd_float2 jitter;
  simd_float2 padding; // 16-byte aligned
};

struct CharacterGPUData {
  int numCharacters;
  int padding[3];
  simd_float4x4 invBoundingBoxes[MAX_CHARACTERS];
  // 6 body parts (head, trunk, left/right arm, left/right leg) per character
  simd_float4x4 invBodyParts[MAX_CHARACTERS * 6]; 
};

struct FrameData {
  simd_float3 sunDirection;
  float time;
  float deltaTime;

  // World origin for toroidal wrapping (world-space coord of indirection cell
  // (0,0,0))
  simd_int3 worldOrigin;
};

struct ExposureData {
  float sceneLuminance; // The smoothed average luminance of the scene
  float padding[3];
};

// Sector handle and flag values are now defined in SystemConfig.h
// SECTOR_HANDLE_EMPTY, SECTOR_HANDLE_LOD, SECTOR_FLAG_DETAIL, SECTOR_FLAG_LOD

struct SectorInfo {
  // Offset into the Brick Arrays (Occupancy and Data)
  // We store the INDEX of the brick, not byte offset.
  uint32_t baseBrickIndex;

  // Flags: 0 = full detail, 1 = LOD (brickMask only, no brick data allocated)
  uint32_t flags;

  // Mask of which 8x8x8 bricks exist in this 32x32x32 sector.
  // 64 bits = 4x4x4 bricks.
  uint64_t brickMask;
};

// Struct to tell the GPU which brick to generate
struct BrickWorkItem {
  uint32_t sectorIndex;     // Index into the Sector Buffer
  uint32_t localBrickIndex; // 0..63 inside that sector
  uint64_t
      occupancyOffset; // Global offset into Occupancy Buffer (index, not byte)
  uint64_t dataOffset; // Global offset into Data Buffer (index, not byte)
};

// Struct for incremental sector analysis work-list
struct SectorWorkItem {
  int32_t worldX; // World-space sector coordinate
  int32_t worldY;
  int32_t worldZ;
  uint32_t wrappedIdx; // Linear index in the indirection texture (wrapped)
};

#endif