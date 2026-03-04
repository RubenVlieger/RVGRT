#pragma once

#include "ShaderTypes.h"
#include "renderer/ShaderTypes.h"
#include "renderer/hitInfo.h"
#include "tables.h"
#include <metal_stdlib>

using namespace metal;

inline uint GetLinearIndex4(uint3 p) {
  return (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4);
}

// Cheaper approximation of 64-bit bit tests.
inline bool BitTestHalf64(ulong value, uint shift, uint mask) {
  uint low = shift < 32 ? uint(value) : uint(value >> 32);
  return (low >> (shift & 31) & mask) != 0;
}

inline int GetIsotropicLOD(ulong mask, uint idx) {
  if (mask == 0) {
    return 4;
  }
  uint currHalf = idx < 32 ? uint(mask) : uint(mask >> 32);
  if ((currHalf >> (idx & 0x0Au) & 0x00330033u) == 0) {
    return 2;
  }
  return 1;
}

// 64-bit Population Count
inline int popcnt64(ulong mask) { return popcount(mask); }

// Prefix popcount: count bits set below the given index
inline uint prefix_popcnt64(ulong mask, uint width) {
  uint lo = uint(mask);
  uint count = 0;

  if (width >= 32) {
    count = popcount(lo);
    lo = uint(mask >> 32);
  }
  uint m = 1u << (width & 31u);
  count += popcount(lo & (m - 1u));
  return count;
}

// =================================================================================
// GEOMETRY INTERSECTIONS
// =================================================================================

// Robust Ray-AABB intersection. Returns the entry point clamped to the box.
inline float3 ClipRayToAABB(float3 origin, float3 dir, float3 invDir,
                            float3 boxMin, float3 boxMax) {
  float3 t0 = (boxMin - origin) * invDir;
  float3 t1 = (boxMax - origin) * invDir;
  float3 tmin = min(t0, t1);
  float3 tmax = max(t0, t1);

  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar = min(min(tmax.x, tmax.y), tmax.z);

  if (tNear <= tFar && tFar > 0) {
    return origin + dir * max(tNear, 0.0f);
  }
  return origin;
}

// Aligns ipos to the cell boundary in the direction the ray is travelling.
// For positive direction: align to upper boundary (ipos | cellMask)
// For negative direction: align to lower boundary (ipos & ~cellMask)
// This is MUCH cheaper than parametric StepPastCell (pure integer ops).
inline void AlignToCellBoundaries(thread int3 &ipos, float3 dir, int lod) {
  int cellMask = lod - 1;
  ipos.x = (dir.x < 0) ? (ipos.x & ~cellMask) : (ipos.x | cellMask);
  ipos.y = (dir.y < 0) ? (ipos.y & ~cellMask) : (ipos.y | cellMask);
  ipos.z = (dir.z < 0) ? (ipos.z & ~cellMask) : (ipos.z | cellMask);
}

/**
 * Checks the hierarchy at the current voxel position `ipos`.
 * If a voxel is found, returns false (hit candidate) and fills outMatID.
 * If empty, aligns `ipos` to the cell boundary using AlignToCellBoundaries.
 * Returns true if empty (stepped), false if solid (hit).
 *
 * The outer ray-march loop handles actual position advancement.
 */
inline bool GetStepPos(thread int3 &ipos, float3 dir,
                       texture3d<uint, access::read> indirection,
                       device SectorInfo *sectors, device ulong *occupancy,
                       device uchar *data, device ulong *sectorMasks,
                       thread uint8_t &outMatID) {
  // Guard: Negative coordinates
  if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  uint3 pos = uint3(ipos);

  // --- Level 1: Sector lookup (32x32x32) ---
  uint3 sectorPos = pos >> 5;

  uint indW = indirection.get_width();
  uint indH = indirection.get_height();
  uint indD = indirection.get_depth();

  if (sectorPos.x >= indW || sectorPos.y >= indH || sectorPos.z >= indD) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  uint sectorIndex = indirection.read(sectorPos).r;

  // Empty sector? Check super-sector mask for bigger skip.
  if (sectorIndex == 0) {
#if 0 // Sector masks disabled for testing
    if (sectorMasks != nullptr) {
      uint3 superPos = sectorPos >> 2; // 4x4x4 sectors per super-sector
      uint superSectorsX = (indW + 3) / 4;
      uint superSectorsZ = (indD + 3) / 4;
      uint superIdx = superPos.x + superPos.z * superSectorsX +
                       superPos.y * superSectorsX * superSectorsZ;

      ulong superMask = sectorMasks[superIdx];

      uint3 localSector = sectorPos & 3;
      uint localIdx = GetLinearIndex4(localSector);

      uint dirOctant = (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) +
                        (dir.z >= 0 ? 4u : 0u);
      ulong maskedSuper =
          superMask & RayMaskOptimizationLUT[localIdx + dirOctant * 64];
      int superLod = GetIsotropicLOD(maskedSuper, localIdx);

      // Scale: each super-sector cell = 32 voxels
      AlignToCellBoundaries(ipos, dir, superLod * 32);
    } else {
      AlignToCellBoundaries(ipos, dir, 32);
    }
#else
    AlignToCellBoundaries(ipos, dir, 32);
#endif
    return true;
  }

  // Load sector data (1-based indexing)
  SectorInfo sec = sectors[sectorIndex];
  uint64_t brickMask = sec.brickMask;

  // --- Level 2: Brick lookup (8x8x8) ---
  uint3 brickRel = (pos >> 3) & 3; // position within sector's 4x4x4 brick grid
  uint brickLinearIdx = GetLinearIndex4(brickRel);

  // Ray direction octant for LUT (positive dir = 1 bit)
  uint dirOctant =
      (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) + (dir.z >= 0 ? 4u : 0u);

  if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
    // Brick exists — drill down to voxel level

    // Calculate which brick this is in the packed array
    uint packedBrickOffset = prefix_popcnt64(brickMask, brickLinearIdx);
    uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;

    // --- Level 3: Sub-brick (4x4x4) occupancy mask ---
    uint3 subPos = (pos >> 2) & 1;
    uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4); // XZY order

    ulong voxMask = occupancy[occIndexBase + subIdx];

    // --- Level 4: Voxel (1x1x1) ---
    uint3 vRel = pos & 3;
    uint vIdx = GetLinearIndex4(vRel);

    // Check if this specific voxel is solid
    if (BitTestHalf64(voxMask, vIdx, 1)) {
      // HIT! Read material data.
      if (data != nullptr) {
        uint dataIdx = (sec.baseBrickIndex + packedBrickOffset) * 512 +
                       (subIdx * 64) + vIdx;
        outMatID = data[dataIdx];
      } else {
        outMatID = 1;
      }
      return false; // Hit
    }

    // Empty voxel inside a solid brick.
    // Apply ray mask LUT and determine LOD for skip distance.
    ulong maskedOcc = voxMask & RayMaskOptimizationLUT[vIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedOcc, vIdx);

    AlignToCellBoundaries(ipos, dir, lod); // 1, 2, or 4 voxels
    return true;

  } else {
    // Empty brick. Apply ray mask LUT at brick scale and determine LOD.
    ulong maskedBrick =
        brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);

    // Scale by brick size (8 voxels per brick)
    AlignToCellBoundaries(ipos, dir, lod * 8); // 8, 16, or 32 voxels
    return true;
  }
}

// =================================================================================
// MAIN TRACE FUNCTIONS
// =================================================================================

inline hitInfo trace(float3 rayPos, float3 rayDir,
                     texture3d<uint, access::read> indirection,
                     device SectorInfo *sectors, device ulong *occupancy,
                     device uchar *data, device ulong *sectorMasks) {
  hitInfo hit;
  hit.hit = false;
  hit.normal = half3(0, 1, 0);
  hit.its = 0;

  // Safe direction: avoid division by zero
  float3 safeDir = rayDir;
  safeDir.x = (abs(safeDir.x) < 1e-8f) ? copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (abs(safeDir.y) < 1e-8f) ? copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (abs(safeDir.z) < 1e-8f) ? copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = 1.0f / safeDir;

  // World bounds from indirection texture
  float3 worldSize = float3(indirection.get_width(), indirection.get_height(),
                            indirection.get_depth()) *
                     32.0f;

  // Clip ray to world AABB
  float3 startPos =
      ClipRayToAABB(rayPos, safeDir, invDir, float3(0), worldSize);

  // tStart = distance from rayPos to the entry face of the first voxel
  float3 tStart;
  tStart.x =
      ((safeDir.x >= 0 ? 1.0f : 0.0f) - (startPos.x - floor(startPos.x))) *
      invDir.x;
  tStart.y =
      ((safeDir.y >= 0 ? 1.0f : 0.0f) - (startPos.y - floor(startPos.y))) *
      invDir.y;
  tStart.z =
      ((safeDir.z >= 0 ? 1.0f : 0.0f) - (startPos.z - floor(startPos.z))) *
      invDir.z;

  float3 currPos = startPos + safeDir * 0.001f;
  float3 sideDist = float3(0.0f);
  float3 worldOrigin = floor(startPos); // Integer reference for tStart math

  // Traversal loop
  int maxIters = 512;

  for (int i = 0; i < maxIters; ++i) {
    hit.its++;

    int3 voxelPos = int3(floor(currPos));

    // Bounds check
    if (voxelPos.x < 0 || voxelPos.y < 0 || voxelPos.z < 0 ||
        voxelPos.x >= int(worldSize.x) || voxelPos.y >= int(worldSize.y) ||
        voxelPos.z >= int(worldSize.z)) {
      break;
    }

    uint8_t matID = 0;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, matID);

    if (!stepped) {
      // Hit a solid voxel — compute hit info from sideDist
      hit.hit = true;

      // Determine which axis face was crossed
      // Use sideDist to determine the entry face
      float3 cellMin = float3(voxelPos);
      float3 t0 = (cellMin - rayPos) * invDir;
      float3 t1 = (cellMin + 1.0f - rayPos) * invDir;
      float3 tmin_v = min(t0, t1);
      float tEntry = max(max(tmin_v.x, tmin_v.y), tmin_v.z);

      hit.pos = rayPos + rayDir * tEntry;

      // Normal from dominant axis of entry
      float3 center = cellMin + 0.5f;
      float3 d = hit.pos - center;
      float3 ad = abs(d);

      if (ad.x > ad.y && ad.x > ad.z)
        hit.normal = half3(sign(d.x), 0, 0);
      else if (ad.y > ad.z)
        hit.normal = half3(0, sign(d.y), 0);
      else
        hit.normal = half3(0, 0, sign(d.z));

      // UV mapping
      float3 fpos = floor(hit.pos);
      float3 localPos = hit.pos - fpos;
      if (abs(hit.normal.x) > 0.5h)
        hit.uv = half2(localPos.y, localPos.z);
      else if (abs(hit.normal.y) > 0.5h)
        hit.uv = half2(localPos.x, localPos.z);
      else
        hit.uv = half2(localPos.x, localPos.y);

      hit.matID = matID;
      return hit;
    }

    // Ray-march advancement: compute sideDist from aligned position
    // voxelPos has been aligned to cell boundaries by GetStepPos
    // Must use floor(startPos) as reference — tStart is defined relative to it
    float3 alignedF = float3(voxelPos) - worldOrigin;
    sideDist.x = tStart.x + alignedF.x * invDir.x;
    sideDist.y = tStart.y + alignedF.y * invDir.y;
    sideDist.z = tStart.z + alignedF.z * invDir.z;

    float tmin = min(min(sideDist.x, sideDist.y), sideDist.z) + 0.001f;
    currPos = startPos + tmin * safeDir;
  }

  return hit;
}

// Optimized Shadow Trace (Boolean)
inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, texture3d<uint, access::read> indirection,
                        device SectorInfo *sectors, device ulong *occupancy,
                        device uchar *data, device ulong *sectorMasks) {
  float3 safeDir = rayDir;
  safeDir.x = (abs(safeDir.x) < 1e-8f) ? copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (abs(safeDir.y) < 1e-8f) ? copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (abs(safeDir.z) < 1e-8f) ? copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = 1.0f / safeDir;

  float3 worldSize = float3(indirection.get_width(), indirection.get_height(),
                            indirection.get_depth()) *
                     32.0f;

  float3 startPos =
      ClipRayToAABB(rayPos, safeDir, invDir, float3(0), worldSize);

  float3 tStart;
  tStart.x =
      ((safeDir.x >= 0 ? 1.0f : 0.0f) - (startPos.x - floor(startPos.x))) *
      invDir.x;
  tStart.y =
      ((safeDir.y >= 0 ? 1.0f : 0.0f) - (startPos.y - floor(startPos.y))) *
      invDir.y;
  tStart.z =
      ((safeDir.z >= 0 ? 1.0f : 0.0f) - (startPos.z - floor(startPos.z))) *
      invDir.z;

  float3 currPos = startPos + safeDir * 0.001f;
  float3 sideDist = float3(0.0f);
  float3 worldOrigin = floor(startPos); // Integer reference for tStart math

  float maxDistSq = maxDist * maxDist;

  for (int i = 0; i < maxIters; ++i) {
    int3 voxelPos = int3(floor(currPos));

    // Bounds check
    if (voxelPos.x < 0 || voxelPos.y < 0 || voxelPos.z < 0 ||
        voxelPos.x >= int(worldSize.x) || voxelPos.y >= int(worldSize.y) ||
        voxelPos.z >= int(worldSize.z)) {
      return false;
    }

    uint8_t matID = 0;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, matID);

    if (!stepped) {
      return true; // Hit occluder
    }

    // Distance cap
    float3 diff = float3(voxelPos) + 0.5f - rayPos;
    if (dot(diff, diff) > maxDistSq) {
      return false;
    }

    // Ray-march advancement
    // Must use floor(startPos) as reference — tStart is defined relative to it
    float3 alignedF = float3(voxelPos) - worldOrigin;
    sideDist.x = tStart.x + alignedF.x * invDir.x;
    sideDist.y = tStart.y + alignedF.y * invDir.y;
    sideDist.z = tStart.z + alignedF.z * invDir.z;

    float tmin = min(min(sideDist.x, sideDist.y), sideDist.z) + 0.001f;
    currPos = startPos + tmin * safeDir;
  }

  return false;
}

inline hitInfo trace(float3 rayPos, float3 rayDir,
                     texture3d<uint, access::read> indirection,
                     device SectorInfo *sectors, device ulong *occupancy,
                     device uchar *data) {
  return trace(rayPos, rayDir, indirection, sectors, occupancy, data, nullptr);
}

inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, texture3d<uint, access::read> indirection,
                        device SectorInfo *sectors, device ulong *occupancy,
                        device uchar *data) {
  return traceShadow(rayPos, rayDir, maxDist, maxIters, indirection, sectors,
                     occupancy, data, nullptr);
}