#ifndef ShaderTypes_h
#define ShaderTypes_h

#if defined(__APPLE__) || defined(__METAL_VERSION__)
// The simd library provides C++ types like `simd_float3` that are
// directly memory-compatible with Metal's `float3`.
#include <simd/simd.h>
#else
// On Windows/CUDA, use the cross-platform math types from cumath.h
// and provide simd_xxx aliases for source compatibility.
#include "cumath.h"
#include <cstdint>

using simd_float2 = float2;
using simd_float3 = float3;
using simd_float4 = float4;
using simd_int3   = int3;
using simd_float4x4 = mat4;

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

#define MAX_CHARACTERS 16

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

#define BRICK_SIZE 8
#define SECTOR_SIZE 32 // 4 bricks * 8 voxels

// =========================================================
// Streaming Configuration
// =========================================================

// Brick pool capacity — how many 8x8x8 bricks can exist simultaneously.
// Each brick uses 576 bytes (64 occupancy + 512 data).
// At 6M bricks: ~3.3GB total.
#define BRICK_POOL_CAPACITY (6 * 1024 * 1024)

// Maximum number of sectors that can be active simultaneously.
// Must be >= the total cells in the indirection texture.
// 256 * 16 * 256 = 1,048,576 sectors.
#define MAX_ACTIVE_SECTORS (256 * 16 * 256)

// Radius (in sectors) within which full-detail bricks are generated.
// 125 sectors * 32 voxels = 4000 blocks.
#define DETAIL_RADIUS_SECTORS 125

// =========================================================
// Sector Handle Sentinel Values
// Stored in the indirection 3D texture (R32Uint).
// =========================================================
// 0 = empty (no geometry, not yet loaded or truly empty)
#define SECTOR_HANDLE_EMPTY 0u
// 0xFFFFFFFE = LOD solid (all bricks treated as solid based on brickMask, no
// brick data)
#define SECTOR_HANDLE_LOD 0xFFFFFFFEu
// Valid sector handles: 1..0xFFFFFFFD (index into SectorBuffer)

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

// Flag values for SectorInfo.flags
#define SECTOR_FLAG_DETAIL 0u
#define SECTOR_FLAG_LOD 1u

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