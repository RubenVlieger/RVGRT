#ifndef ShaderTypes_h
#define ShaderTypes_h

// The simd library provides C++ types like `simd_float3` that are
// directly memory-compatible with Metal's `float3`.
#include <simd/simd.h>

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

struct FrameData {
  simd_float3 sunDirection;
  float time;
  float deltaTime;
};

struct ExposureData {
  float sceneLuminance; // The smoothed average luminance of the scene
  float padding[3];
};

#define BRICK_SIZE 8
#define SECTOR_SIZE 32 // 4 bricks * 8 voxels

struct SectorInfo {
  // Offset into the Brick Arrays (Occupancy and Data)
  // We store the INDEX of the brick, not byte offset.
  uint32_t baseBrickIndex;

  uint32_t padding; // Align to 8 bytes for the uint64 following

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

#endif