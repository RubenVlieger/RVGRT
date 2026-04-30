#pragma once
#include "SystemConfig.h"

#ifndef SHADER_TYPES_MATH_INCLUDED
#error "ShaderTypesMath.h must be included before ShaderTypesCore.h"
#endif

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
  simd_float4 characterCenters[MAX_CHARACTERS];
  simd_float4x4 invBoundingBoxes[MAX_CHARACTERS];
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

struct GlyphInstance {
    simd_float2 screenPos;     // Top-left corner in screen pixels
    simd_float2 screenSize;    // Width/height in pixels of this glyph instance
    simd_float2 atlasUVMin;     // Top-left UV in the SDF atlas
    simd_float2 atlasUVMax;     // Bottom-right UV in the SDF atlas
    simd_float4 color;          // RGBA tint (premultiplied alpha)
    float softness;             // SDF edge softness for anti-aliasing
    float sceneDepth;           // Depth for 3D text occlusion (FLT_MAX for HUD)
    uint32_t flags;             // Bit 0: depth test enable, Bit 1: solid rect (no SDF)
    uint32_t _pad;
};

struct TextOverlayData {
    uint32_t numGlyphs;
    uint32_t numTilesX;
    uint32_t numTilesY;
    uint32_t screenWidth;
    uint32_t screenHeight;
};
