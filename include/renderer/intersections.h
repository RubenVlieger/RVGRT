#pragma once

#include "ShaderTypes.h"
#include "renderer/ShaderTypes.h"
#include "renderer/shader_settings.h"
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
inline void AlignToCellBoundaries(thread int3 &ipos, float3 dir, int lod) {
  int cellMask = lod - 1;
  ipos.x = (dir.x < 0) ? (ipos.x & ~cellMask) : (ipos.x | cellMask);
  ipos.y = (dir.y < 0) ? (ipos.y & ~cellMask) : (ipos.y | cellMask);
  ipos.z = (dir.z < 0) ? (ipos.z & ~cellMask) : (ipos.z | cellMask);
}

/**
 * Checks the hierarchy at the current voxel position `ipos`.
 * Supports toroidal wrapping via worldOrigin and LOD sectors.
 *
 * worldOrigin: world-space sector coordinate of indirection cell (0,0,0).
 *              Voxel at world position `v` maps to indirection cell
 *              `(v/32 - worldOrigin) mod indirectionSize`.
 */
inline bool GetStepPos(thread int3 &ipos, float3 dir,
                       texture3d<uint, access::read> indirection,
                       device SectorInfo *sectors, device ulong *occupancy,
                       device uchar *data, device ulong *sectorMasks,
                       int3 worldOrigin, thread uint8_t &outMatID) {
  // Y bounds check (height is hard-limited)
  if (ipos.y < 0 || ipos.y >= int(SIZEY)) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  uint3 pos = uint3(ipos.x, ipos.y, ipos.z);

  // --- Level 1: Sector lookup (32x32x32) ---
  // World sector position
  int3 worldSector = int3(ipos.x >> 5, ipos.y >> 5, ipos.z >> 5);

  // Toroidal wrapping: convert world sector to indirection cell
  uint indW = indirection.get_width();
  uint indH = indirection.get_height();
  uint indD = indirection.get_depth();

  // Wrap to indirection texture coordinates
  int3 relSector = worldSector - worldOrigin;

  // Check if within the loaded region
  if (relSector.x < 0 || relSector.x >= int(indW) || relSector.y < 0 ||
      relSector.y >= int(indH) || relSector.z < 0 || relSector.z >= int(indD)) {
    // Outside loaded region — treat as air
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  // Evaluate positive modulo for toroidal wrap lookup in the 3D texture
  uint wx = (worldSector.x % int(indW) + int(indW)) % int(indW);
  uint wy = (worldSector.y % int(indH) + int(indH)) % int(indH);
  uint wz = (worldSector.z % int(indD) + int(indD)) % int(indD);
  uint3 sectorPos = uint3(wx, wy, wz);
  uint sectorIndex = indirection.read(sectorPos).r;

  // Empty sector
  if (sectorIndex == SECTOR_HANDLE_EMPTY) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  // Load sector data
  SectorInfo sec = sectors[sectorIndex];
  uint64_t brickMask = sec.brickMask;

  // --- Level 2: Brick lookup (8x8x8) ---
  uint3 localPos =
      uint3(uint(ipos.x) & 31u, uint(ipos.y) & 31u, uint(ipos.z) & 31u);
  uint3 brickRel =
      (localPos >> 3) & 3; // position within sector's 4x4x4 brick grid
  uint brickLinearIdx = GetLinearIndex4(brickRel);

  // Ray direction octant for LUT
  uint dirOctant =
      (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) + (dir.z >= 0 ? 4u : 0u);

  // --- LOD SECTOR HANDLING ---
  if (sec.flags == SECTOR_FLAG_LOD) {
    // LOD: brickMask tells us which 8x8x8 bricks are solid/air
    if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
      // LOD brick is "solid" — return a hit with generic stone material
      outMatID = MAT_STONE;
      return false; // Hit
    } else {
      // LOD brick is "air" — skip using ray mask LUT
      ulong maskedBrick =
          brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
      int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
      AlignToCellBoundaries(ipos, dir, lod * 8);
      return true;
    }
  }

  // --- FULL DETAIL PATH ---
  if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
    // Brick exists — drill down to voxel level
    uint packedBrickOffset = prefix_popcnt64(brickMask, brickLinearIdx);
    uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;

    // --- Level 3: Sub-brick (4x4x4) occupancy mask ---
    uint3 subPos = (localPos >> 2) & 1;
    uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4); // XZY order

    ulong voxMask = occupancy[occIndexBase + subIdx];

    // --- Level 4: Voxel (1x1x1) ---
    uint3 vRel = localPos & 3;
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
    ulong maskedOcc = voxMask & RayMaskOptimizationLUT[vIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedOcc, vIdx);

    AlignToCellBoundaries(ipos, dir, lod);
    return true;

  } else {
    // Empty brick. Apply ray mask LUT at brick scale.
    ulong maskedBrick =
        brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);

    AlignToCellBoundaries(ipos, dir, lod * 8);
    return true;
  }
}

// =================================================================================
// MAIN TRACE FUNCTIONS (with worldOrigin for streaming)
// =================================================================================

inline hitInfo trace(float3 rayPos, float3 rayDir,
                     texture3d<uint, access::read> indirection,
                     device SectorInfo *sectors, device ulong *occupancy,
                     device uchar *data, device ulong *sectorMasks,
                     int3 worldOrigin,
                     constant CharacterGPUData* charData = nullptr) {
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

  // World bounds: defined by the loaded region around the world origin
  float3 worldMin = float3(worldOrigin) * 32.0f;
  float3 worldMax = float3(worldOrigin.x + int(indirection.get_width()),
                           worldOrigin.y + int(indirection.get_height()),
                           worldOrigin.z + int(indirection.get_depth())) *
                    32.0f;

  // Clip ray to world AABB
  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);

  int3 voxelPos = int3(floor(startPos));
  
  // ---------- Character AABB intersections (O(1) closest test) ----------
  float closestCharT = 1e20f;
  float3 closestCharNormal = float3(0.0f);
  int closestCharMatID = 0;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          // Check overall bounding box first
          float4x4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = (invBBox * float4(rayDir, 0.0f)).xyz;
          float3 localBPos = (invBBox * float4(rayPos, 1.0f)).xyz;
          float3 invLocalBDir = 1.0f / localBDir;
          
          float3 bt0 = (-0.5f - localBPos) * invLocalBDir;
          float3 bt1 = ( 0.5f - localBPos) * invLocalBDir;
          float3 btmin = min(bt0, bt1);
          float3 btmax = max(bt0, bt1);
          float bNear = max(max(btmin.x, btmin.y), btmin.z);
          float bFar = min(min(btmax.x, btmax.y), btmax.z);
          
          // Self-intersection check: if ray origin is inside bounding box (bNear < 0 AND bFar > 0),
          // we are generating primary rays from inside our own head, so we should skip this character entirely 
          // in the primary trace trace() to avoid black/flickering screens blocking the view.
          if (bNear < 0.0f && bFar > 0.0f) continue;
          
          // Missed the bounding box entirely, or it's behind us, or further than a found character hit
          if (bNear > bFar || bFar < 0.0f || bNear >= closestCharT) continue;
          
          // Bounding box hit! Now check the 6 body parts
          for (int p = 0; p < 6; ++p) {
              float4x4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = (invPart * float4(rayDir, 0.0f)).xyz;
              float3 localPos = (invPart * float4(rayPos, 1.0f)).xyz;
              float3 invLocalDir = 1.0f / localDir;
              
              float3 t0 = (-0.5f - localPos) * invLocalDir;
              float3 t1 = ( 0.5f - localPos) * invLocalDir;
              float3 tmin = min(t0, t1);
              float3 tmax = max(t0, t1);
              float tNear = max(max(tmin.x, tmin.y), tmin.z);
              float tFar = min(min(tmax.x, tmax.y), tmax.z);
              
              if (tNear <= tFar && tFar > 0.0f && tNear < closestCharT) {
                  closestCharT = max(tNear, 0.0f);
                  closestCharMatID = 255;
                  
                  // Calculate local normal of the AABB
                  float3 hitPosLocal = localPos + localDir * closestCharT;
                  float3 absHit = abs(hitPosLocal);
                  float3 localNormal = float3(0.0f);
                  if (absHit.x > absHit.y && absHit.x > absHit.z)
                      localNormal = float3(sign(hitPosLocal.x), 0.0f, 0.0f);
                  else if (absHit.y > absHit.z)
                      localNormal = float3(0.0f, sign(hitPosLocal.y), 0.0f);
                  else
                      localNormal = float3(0.0f, 0.0f, sign(hitPosLocal.z));
                      
                  // Transform normal back to world space: n_world = transpose(inverse(M)) * n_local
                  // wait! invPart IS the inverse! So we want transpose(invPart) * n_local
                  // Note: metal float4x4 matrix multiply with float3 vector implicitly pads w=0
                  closestCharNormal = normalize((transpose(invPart) * float4(localNormal, 0.0f)).xyz);
              }
          }
      }
      #endif
  }
  // ----------------------------------------------------------------------

  // Traversal loop
  int maxIters = 512;

  for (int i = 0; i < maxIters; ++i) {
    hit.its++;

    // Bounds check (using loaded region)
    if (float(voxelPos.x) < worldMin.x || float(voxelPos.y) < worldMin.y ||
        float(voxelPos.z) < worldMin.z || float(voxelPos.x) >= worldMax.x ||
        float(voxelPos.y) >= worldMax.y || float(voxelPos.z) >= worldMax.z) {
      break;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID);

    if (!stepped) {
      hit.hit = true;

      float3 cellMin = float3(voxelPos);
      float3 t0 = (cellMin - rayPos) * invDir;
      float3 t1 = (cellMin + 1.0f - rayPos) * invDir;
      float3 tmin_v = min(t0, t1);
      float tEntry = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
      
      // Compare voxel intersection distance vs character intersection
      if (closestCharT < tEntry) {
          hit.pos = rayPos + rayDir * closestCharT;
          hit.normal = half3(closestCharNormal);
          hit.matID = closestCharMatID;
          hit.uv = half2(0.0h);
          return hit;
      }

      hit.pos = rayPos + rayDir * tEntry;

      float3 center = cellMin + 0.5f;
      float3 d = hit.pos - center;
      float3 ad = abs(d);

      if (ad.x > ad.y && ad.x > ad.z)
        hit.normal = half3(sign(d.x), 0, 0);
      else if (ad.y > ad.z)
        hit.normal = half3(0, sign(d.y), 0);
      else
        hit.normal = half3(0, 0, sign(d.z));

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

    float3 nextBoundary;
    nextBoundary.x = (safeDir.x >= 0) ? (float(voxelPos.x) + 1.0f) : float(voxelPos.x);
    nextBoundary.y = (safeDir.y >= 0) ? (float(voxelPos.y) + 1.0f) : float(voxelPos.y);
    nextBoundary.z = (safeDir.z >= 0) ? (float(voxelPos.z) + 1.0f) : float(voxelPos.z);

    float3 tMax;
    tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
    tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
    tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

    float tmin = min(min(tMax.x, tMax.y), tMax.z);

    int3 nextVoxelPos;
    nextVoxelPos.x = int(floor(startPos.x + tmin * safeDir.x));
    nextVoxelPos.y = int(floor(startPos.y + tmin * safeDir.y));
    nextVoxelPos.z = int(floor(startPos.z + tmin * safeDir.z));

    // Clamping to avoid backwards evaluation drift from float precision limits
    nextVoxelPos.x = (safeDir.x >= 0) ? max(nextVoxelPos.x, originalVoxelPos.x) : min(nextVoxelPos.x, originalVoxelPos.x);
    nextVoxelPos.y = (safeDir.y >= 0) ? max(nextVoxelPos.y, originalVoxelPos.y) : min(nextVoxelPos.y, originalVoxelPos.y);
    nextVoxelPos.z = (safeDir.z >= 0) ? max(nextVoxelPos.z, originalVoxelPos.z) : min(nextVoxelPos.z, originalVoxelPos.z);

    // Explicitly step the exact integer crossed boundary
    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
        nextVoxelPos.x = voxelPos.x + (safeDir.x >= 0 ? 1 : -1);
    } else if (tMax.y <= tMax.z) {
        nextVoxelPos.y = voxelPos.y + (safeDir.y >= 0 ? 1 : -1);
    } else {
        nextVoxelPos.z = voxelPos.z + (safeDir.z >= 0 ? 1 : -1);
    }

    voxelPos = nextVoxelPos;
  }

  if (closestCharT < 1e20f) {
      hit.hit = true;
      hit.pos = rayPos + rayDir * closestCharT;
      hit.normal = half3(closestCharNormal);
      hit.matID = closestCharMatID;
      hit.uv = half2(0.0h);
      return hit;
  }

  return hit;
}

// Optimized Shadow Trace (Boolean) with worldOrigin
inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, texture3d<uint, access::read> indirection,
                        device SectorInfo *sectors, device ulong *occupancy,
                        device uchar *data, device ulong *sectorMasks,
                        int3 worldOrigin,
                        constant CharacterGPUData* charData = nullptr) {
  float3 safeDir = rayDir;
  safeDir.x = (abs(safeDir.x) < 1e-8f) ? copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (abs(safeDir.y) < 1e-8f) ? copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (abs(safeDir.z) < 1e-8f) ? copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = 1.0f / safeDir;

  float3 worldMin = float3(worldOrigin) * 32.0f;
  float3 worldMax = float3(worldOrigin.x + int(indirection.get_width()),
                           worldOrigin.y + int(indirection.get_height()),
                           worldOrigin.z + int(indirection.get_depth())) *
                    32.0f;

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);

  float maxDistSq = maxDist * maxDist;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          float4x4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = (invBBox * float4(rayDir, 0.0f)).xyz;
          float3 localBPos = (invBBox * float4(rayPos, 1.0f)).xyz;
          float3 invLocalBDir = 1.0f / localBDir;
          float3 bt0 = (-0.5f - localBPos) * invLocalBDir;
          float3 bt1 = ( 0.5f - localBPos) * invLocalBDir;
          float3 btmin = min(bt0, bt1);
          float3 btmax = max(bt0, bt1);
          float bNear = max(max(btmin.x, btmin.y), btmin.z);
          float bFar = min(min(btmax.x, btmax.y), btmax.z);
          
          if (bNear > bFar || bFar < 0.0f || bNear * bNear > maxDistSq) continue;
          
          for (int p = 0; p < 6; ++p) {
              float4x4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = (invPart * float4(rayDir, 0.0f)).xyz;
              float3 localPos = (invPart * float4(rayPos, 1.0f)).xyz;
              float3 invLocalDir = 1.0f / localDir;
              float3 t0 = (-0.5f - localPos) * invLocalDir;
              float3 t1 = ( 0.5f - localPos) * invLocalDir;
              float3 tmin = min(t0, t1);
              float3 tmax = max(t0, t1);
              float tNear = max(max(tmin.x, tmin.y), tmin.z);
              float tFar = min(min(tmax.x, tmax.y), tmax.z);
              
              if (tNear <= tFar && tFar > 0.0f && (tNear * tNear) < maxDistSq) {
                  return true;
              }
          }
      }
      #endif
  }

  int3 voxelPos = int3(floor(startPos));

  for (int i = 0; i < maxIters; ++i) {

    // Bounds check
    if (float(voxelPos.x) < worldMin.x || float(voxelPos.y) < worldMin.y ||
        float(voxelPos.z) < worldMin.z || float(voxelPos.x) >= worldMax.x ||
        float(voxelPos.y) >= worldMax.y || float(voxelPos.z) >= worldMax.z) {
      return false;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID);

    if (!stepped) {
      return true; // Hit occluder
    }

    float3 diff = float3(voxelPos) + 0.5f - rayPos;
    if (dot(diff, diff) > maxDistSq) {
      return false;
    }

    float3 nextBoundary;
    nextBoundary.x = (safeDir.x >= 0) ? (float(voxelPos.x) + 1.0f) : float(voxelPos.x);
    nextBoundary.y = (safeDir.y >= 0) ? (float(voxelPos.y) + 1.0f) : float(voxelPos.y);
    nextBoundary.z = (safeDir.z >= 0) ? (float(voxelPos.z) + 1.0f) : float(voxelPos.z);

    float3 tMax;
    tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
    tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
    tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

    float tmin = min(min(tMax.x, tMax.y), tMax.z);

    int3 nextVoxelPos;
    nextVoxelPos.x = int(floor(startPos.x + tmin * safeDir.x));
    nextVoxelPos.y = int(floor(startPos.y + tmin * safeDir.y));
    nextVoxelPos.z = int(floor(startPos.z + tmin * safeDir.z));

    nextVoxelPos.x = (safeDir.x >= 0) ? max(nextVoxelPos.x, originalVoxelPos.x) : min(nextVoxelPos.x, originalVoxelPos.x);
    nextVoxelPos.y = (safeDir.y >= 0) ? max(nextVoxelPos.y, originalVoxelPos.y) : min(nextVoxelPos.y, originalVoxelPos.y);
    nextVoxelPos.z = (safeDir.z >= 0) ? max(nextVoxelPos.z, originalVoxelPos.z) : min(nextVoxelPos.z, originalVoxelPos.z);

    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
        nextVoxelPos.x = voxelPos.x + (safeDir.x >= 0 ? 1 : -1);
    } else if (tMax.y <= tMax.z) {
        nextVoxelPos.y = voxelPos.y + (safeDir.y >= 0 ? 1 : -1);
    } else {
        nextVoxelPos.z = voxelPos.z + (safeDir.z >= 0 ? 1 : -1);
    }

    voxelPos = nextVoxelPos;
  }

  return false;
}

// Legacy overloads (without worldOrigin — default to origin 0,0,0)
inline hitInfo trace(float3 rayPos, float3 rayDir,
                     texture3d<uint, access::read> indirection,
                     device SectorInfo *sectors, device ulong *occupancy,
                     device uchar *data, device ulong *sectorMasks) {
  return trace(rayPos, rayDir, indirection, sectors, occupancy, data,
               sectorMasks, int3(0), nullptr);
}

inline hitInfo trace(float3 rayPos, float3 rayDir,
                     texture3d<uint, access::read> indirection,
                     device SectorInfo *sectors, device ulong *occupancy,
                     device uchar *data) {
  return trace(rayPos, rayDir, indirection, sectors, occupancy, data, nullptr,
               int3(0), nullptr);
}

inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, texture3d<uint, access::read> indirection,
                        device SectorInfo *sectors, device ulong *occupancy,
                        device uchar *data, device ulong *sectorMasks) {
  return traceShadow(rayPos, rayDir, maxDist, maxIters, indirection, sectors,
                     occupancy, data, sectorMasks, int3(0), nullptr);
}

inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, texture3d<uint, access::read> indirection,
                        device SectorInfo *sectors, device ulong *occupancy,
                        device uchar *data) {
  return traceShadow(rayPos, rayDir, maxDist, maxIters, indirection, sectors,
                     occupancy, data, nullptr, int3(0), nullptr);
}