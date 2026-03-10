#pragma once

// ============================================================================
// CROSS-PLATFORM RAY-Voxel INTERSECTION
// 
// This header provides unified ray tracing for voxel SVO data structures.
// Works on Metal, CUDA, and C++ with identical logic.
// ============================================================================

#if defined(__METAL_VERSION__)
// ============================================================================
// METAL IMPLEMENTATION
// ============================================================================
#include "ShaderTypes.h"
#include "renderer/ShaderTypes.h"
#include "renderer/shader_settings.h"
#include "renderer/hitInfo.h"
#include "tables.h"
#include <metal_stdlib>

using namespace metal;

// Type aliases for Metal
#define SVO_TEXTURE_TYPE texture3d<uint, access::read>
#define SECTOR_BUFFER_TYPE device SectorInfo*
#define OCCUPANCY_TYPE device ulong*
#define DATA_TYPE device uchar*
#define MASK_TYPE device ulong*
#define CHAR_DATA_TYPE constant CharacterGPUData*

inline uint GetLinearIndex4(uint3 p) {
  return (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4);
}

inline bool BitTestHalf64(ulong value, uint shift, uint mask) {
  uint low = shift < 32 ? uint(value) : uint(value >> 32);
  return (low >> (shift & 31) & mask) != 0;
}

inline int GetIsotropicLOD(ulong mask, uint idx) {
  if (mask == 0) return 4;
  uint currHalf = idx < 32 ? uint(mask) : uint(mask >> 32);
  if ((currHalf >> (idx & 0x0Au) & 0x00330033u) == 0) return 2;
  return 1;
}

inline int popcnt64(ulong mask) { return popcount(mask); }

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

inline void AlignToCellBoundaries(thread int3 &ipos, float3 dir, int lod) {
  int cellMask = lod - 1;
  ipos.x = (dir.x < 0) ? (ipos.x & ~cellMask) : (ipos.x | cellMask);
  ipos.y = (dir.y < 0) ? (ipos.y & ~cellMask) : (ipos.y | cellMask);
  ipos.z = (dir.z < 0) ? (ipos.z & ~cellMask) : (ipos.z | cellMask);
}

inline uint ReadIndirection(SVO_TEXTURE_TYPE indirection, uint3 pos, 
                            int indW, int indH, int indD) {
  (void)indW; (void)indH; (void)indD; // Unused in Metal path
  return indirection.read(pos).r;
}

inline bool GetStepPos(thread int3 &ipos, float3 dir,
                       SVO_TEXTURE_TYPE indirection,
                       SECTOR_BUFFER_TYPE sectors, OCCUPANCY_TYPE occupancy,
                       DATA_TYPE data, MASK_TYPE sectorMasks,
                       int3 worldOrigin, thread uint8_t &outMatID,
                       int indW, int indH, int indD) {
  if (ipos.y < 0 || ipos.y >= int(SIZEY)) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  int3 worldSector = int3(ipos.x >> 5, ipos.y >> 5, ipos.z >> 5);
  int3 relSector = worldSector - worldOrigin;

  if (relSector.x < 0 || relSector.x >= indW || relSector.y < 0 ||
      relSector.y >= indH || relSector.z < 0 || relSector.z >= indD) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  uint wx = (worldSector.x % indW + indW) % indW;
  uint wy = (worldSector.y % indH + indH) % indH;
  uint wz = (worldSector.z % indD + indD) % indD;
  uint3 sectorPos = uint3(wx, wy, wz);
  uint sectorIndex = ReadIndirection(indirection, sectorPos, indW, indH, indD);

  if (sectorIndex == SECTOR_HANDLE_EMPTY) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  SectorInfo sec = sectors[sectorIndex];
  uint64_t brickMask = sec.brickMask;

  uint3 localPos = uint3(uint(ipos.x) & 31u, uint(ipos.y) & 31u, uint(ipos.z) & 31u);
  uint3 brickRel = (localPos >> 3) & 3;
  uint brickLinearIdx = GetLinearIndex4(brickRel);

  uint dirOctant = (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) + (dir.z >= 0 ? 4u : 0u);

  if (sec.flags == SECTOR_FLAG_LOD) {
    if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
      outMatID = MAT_STONE;
      return false;
    } else {
      ulong maskedBrick = brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
      int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
      AlignToCellBoundaries(ipos, dir, lod * 8);
      return true;
    }
  }

  if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
    uint packedBrickOffset = prefix_popcnt64(brickMask, brickLinearIdx);
    uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;

    uint3 subPos = (localPos >> 2) & 1;
    uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4);

    ulong voxMask = occupancy[occIndexBase + subIdx];

    uint3 vRel = localPos & 3;
    uint vIdx = GetLinearIndex4(vRel);

    if (BitTestHalf64(voxMask, vIdx, 1)) {
      if (data != nullptr) {
        uint dataIdx = (sec.baseBrickIndex + packedBrickOffset) * 512 + (subIdx * 64) + vIdx;
        outMatID = data[dataIdx];
      } else {
        outMatID = 1;
      }
      return false;
    }

    ulong maskedOcc = voxMask & RayMaskOptimizationLUT[vIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedOcc, vIdx);
    AlignToCellBoundaries(ipos, dir, lod);
    return true;
  } else {
    ulong maskedBrick = brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
    AlignToCellBoundaries(ipos, dir, lod * 8);
    return true;
  }
}

inline hitInfo trace(float3 rayPos, float3 rayDir,
                     SVO_TEXTURE_TYPE indirection,
                     SECTOR_BUFFER_TYPE sectors, OCCUPANCY_TYPE occupancy,
                     DATA_TYPE data, MASK_TYPE sectorMasks,
                     int3 worldOrigin, int indW, int indH, int indD,
                     CHAR_DATA_TYPE charData = nullptr) {
  hitInfo hit;
  hit.hit = false;
  hit.normal = half3(0, 1, 0);
  hit.its = 0;

  float3 safeDir = rayDir;
  safeDir.x = (abs(safeDir.x) < 1e-8f) ? copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (abs(safeDir.y) < 1e-8f) ? copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (abs(safeDir.z) < 1e-8f) ? copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = 1.0f / safeDir;

  float3 worldMin = float3(worldOrigin) * 32.0f;
  float3 worldMax = float3(worldOrigin.x + indW,
                           worldOrigin.y + indH,
                           worldOrigin.z + indD) * 32.0f;

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
  int3 voxelPos = int3(floor(startPos));
  
  float closestCharT = 1e20f;
  float3 closestCharNormal = float3(0.0f);
  int closestCharMatID = 0;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          float4x4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = (invBBox * float4(rayDir, 0.0f)).xyz;
          float3 localBPos = (invBBox * float4(rayPos, 1.0f)).xyz;
          
          float3 safeLocalBDir = localBDir;
          safeLocalBDir.x = (abs(safeLocalBDir.x) < 1e-8f) ? copysign(1e-8f, safeLocalBDir.x) : safeLocalBDir.x;
          safeLocalBDir.y = (abs(safeLocalBDir.y) < 1e-8f) ? copysign(1e-8f, safeLocalBDir.y) : safeLocalBDir.y;
          safeLocalBDir.z = (abs(safeLocalBDir.z) < 1e-8f) ? copysign(1e-8f, safeLocalBDir.z) : safeLocalBDir.z;
          float3 invLocalBDir = 1.0f / safeLocalBDir;
          
          float3 bt0 = (-0.5f - localBPos) * invLocalBDir;
          float3 bt1 = ( 0.5f - localBPos) * invLocalBDir;
          float3 btmin = min(bt0, bt1);
          float3 btmax = max(bt0, bt1);
          float bNear = max(max(btmin.x, btmin.y), btmin.z);
          float bFar = min(min(btmax.x, btmax.y), btmax.z);
          
          if (bNear < 0.0f && bFar > 0.0f) continue;
          if (bNear > bFar || bFar < 0.0f || bNear >= closestCharT) continue;
          
          for (int p = 0; p < 6; ++p) {
              float4x4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = (invPart * float4(rayDir, 0.0f)).xyz;
              float3 localPos = (invPart * float4(rayPos, 1.0f)).xyz;
              
              float3 safeLocalDir = localDir;
              safeLocalDir.x = (abs(safeLocalDir.x) < 1e-8f) ? copysign(1e-8f, safeLocalDir.x) : safeLocalDir.x;
              safeLocalDir.y = (abs(safeLocalDir.y) < 1e-8f) ? copysign(1e-8f, safeLocalDir.y) : safeLocalDir.y;
              safeLocalDir.z = (abs(safeLocalDir.z) < 1e-8f) ? copysign(1e-8f, safeLocalDir.z) : safeLocalDir.z;
              float3 invLocalDir = 1.0f / safeLocalDir;
              
              float3 t0 = (-0.5f - localPos) * invLocalDir;
              float3 t1 = ( 0.5f - localPos) * invLocalDir;
              float3 tmin = min(t0, t1);
              float3 tmax = max(t0, t1);
              float tNear = max(max(tmin.x, tmin.y), tmin.z);
              float tFar = min(min(tmax.x, tmax.y), tmax.z);
              
              if (tNear <= tFar && tFar > 0.0f && tNear < closestCharT) {
                  closestCharT = max(tNear, 0.0f);
                  closestCharMatID = 255;
                  
                  float3 hitPosLocal = localPos + localDir * closestCharT;
                  float3 absHit = abs(hitPosLocal);
                  float3 localNormal = float3(0.0f);
                  if (absHit.x > absHit.y && absHit.x > absHit.z)
                      localNormal = float3(sign(hitPosLocal.x), 0.0f, 0.0f);
                  else if (absHit.y > absHit.z)
                      localNormal = float3(0.0f, sign(hitPosLocal.y), 0.0f);
                  else
                      localNormal = float3(0.0f, 0.0f, sign(hitPosLocal.z));
                      
                  closestCharNormal = normalize((transpose(invPart) * float4(localNormal, 0.0f)).xyz);
              }
          }
      }
      #endif
  }

  int maxIters = 512;
  for (int i = 0; i < maxIters; ++i) {
    hit.its++;

    if (float(voxelPos.x) < worldMin.x || float(voxelPos.y) < worldMin.y ||
        float(voxelPos.z) < worldMin.z || float(voxelPos.x) >= worldMax.x ||
        float(voxelPos.y) >= worldMax.y || float(voxelPos.z) >= worldMax.z) {
      break;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID,
                              indW, indH, indD);

    if (!stepped) {
      hit.hit = true;

      float3 cellMin = float3(voxelPos);
      float3 t0 = (cellMin - rayPos) * invDir;
      float3 t1 = (cellMin + 1.0f - rayPos) * invDir;
      float3 tmin_v = min(t0, t1);
      float tEntry = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
      
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

inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, SVO_TEXTURE_TYPE indirection,
                        SECTOR_BUFFER_TYPE sectors, OCCUPANCY_TYPE occupancy,
                        DATA_TYPE data, MASK_TYPE sectorMasks,
                        int3 worldOrigin, int indW, int indH, int indD,
                        CHAR_DATA_TYPE charData = nullptr) {
  float3 safeDir = rayDir;
  safeDir.x = (abs(safeDir.x) < 1e-8f) ? copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (abs(safeDir.y) < 1e-8f) ? copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (abs(safeDir.z) < 1e-8f) ? copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = 1.0f / safeDir;

  float3 worldMin = float3(worldOrigin) * 32.0f;
  float3 worldMax = float3(worldOrigin.x + indW,
                           worldOrigin.y + indH,
                           worldOrigin.z + indD) * 32.0f;

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
  float maxDistSq = maxDist * maxDist;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          float4x4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = (invBBox * float4(rayDir, 0.0f)).xyz;
          float3 localBPos = (invBBox * float4(rayPos, 1.0f)).xyz;
          
          float3 safeLocalBDir = localBDir;
          safeLocalBDir.x = (abs(safeLocalBDir.x) < 1e-8f) ? copysign(1e-8f, safeLocalBDir.x) : safeLocalBDir.x;
          safeLocalBDir.y = (abs(safeLocalBDir.y) < 1e-8f) ? copysign(1e-8f, safeLocalBDir.y) : safeLocalBDir.y;
          safeLocalBDir.z = (abs(safeLocalBDir.z) < 1e-8f) ? copysign(1e-8f, safeLocalBDir.z) : safeLocalBDir.z;
          float3 invLocalBDir = 1.0f / safeLocalBDir;
          
          float3 bt0 = (-0.5f - localBPos) * invLocalBDir;
          float3 bt1 = ( 0.5f - localBPos) * invLocalBDir;
          float3 btmin = min(bt0, bt1);
          float3 btmax = max(bt0, bt1);
          float bNear = max(max(btmin.x, btmin.y), btmin.z);
          float bFar = min(min(btmax.x, btmax.y), btmax.z);
          
          float bStartDist = max(0.0f, bNear);
          if (bNear > bFar || bFar < 0.0f || (bStartDist * bStartDist) > maxDistSq) continue;
          
          for (int p = 0; p < 6; ++p) {
              float4x4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = (invPart * float4(rayDir, 0.0f)).xyz;
              float3 localPos = (invPart * float4(rayPos, 1.0f)).xyz;
              
              float3 safeLocalDir = localDir;
              safeLocalDir.x = (abs(safeLocalDir.x) < 1e-8f) ? copysign(1e-8f, safeLocalDir.x) : safeLocalDir.x;
              safeLocalDir.y = (abs(safeLocalDir.y) < 1e-8f) ? copysign(1e-8f, safeLocalDir.y) : safeLocalDir.y;
              safeLocalDir.z = (abs(safeLocalDir.z) < 1e-8f) ? copysign(1e-8f, safeLocalDir.z) : safeLocalDir.z;
              float3 invLocalDir = 1.0f / safeLocalDir;
              
              float3 t0 = (-0.5f - localPos) * invLocalDir;
              float3 t1 = ( 0.5f - localPos) * invLocalDir;
              float3 tmin = min(t0, t1);
              float3 tmax = max(t0, t1);
              float tNear = max(max(tmin.x, tmin.y), tmin.z);
              float tFar = min(min(tmax.x, tmax.y), tmax.z);
              
              float tStartDist = max(0.0f, tNear);
              if (tNear <= tFar && tFar > 0.0f && (tStartDist * tStartDist) < maxDistSq) {
                  return true;
              }
          }
      }
      #endif
  }

  int3 voxelPos = int3(floor(startPos));

  for (int i = 0; i < maxIters; ++i) {
    if (float(voxelPos.x) < worldMin.x || float(voxelPos.y) < worldMin.y ||
        float(voxelPos.z) < worldMin.z || float(voxelPos.x) >= worldMax.x ||
        float(voxelPos.y) >= worldMax.y || float(voxelPos.z) >= worldMax.z) {
      return false;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID,
                              indW, indH, indD);

    if (!stepped) return true;

    float3 diff = float3(voxelPos) + 0.5f - rayPos;
    if (dot(diff, diff) > maxDistSq) return false;

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

#elif defined(__CUDACC__)
// ============================================================================
// CUDA IMPLEMENTATION
// ============================================================================
#include "renderer/ShaderTypes.h"
#include "renderer/shader_settings.h"
#include "renderer/hitInfo.h"
#include "tables.h"
#include <cuda_runtime.h>

// Type aliases for CUDA
#define SVO_TEXTURE_TYPE const uint32_t* __restrict__
#define SECTOR_BUFFER_TYPE const SectorInfo* __restrict__
#define OCCUPANCY_TYPE const uint64_t* __restrict__
#define DATA_TYPE const uint8_t* __restrict__
#define MASK_TYPE const uint64_t* __restrict__
#define CHAR_DATA_TYPE const CharacterGPUData*

// CUDA-specific type mappings
using uint = unsigned int;
using uchar = unsigned char;
using ulong = uint64_t;

__device__ inline uint GetLinearIndex4(uint3 p) {
  return (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4);
}

__device__ inline bool BitTestHalf64(ulong value, uint shift, uint mask) {
  uint low = shift < 32 ? uint(value) : uint(value >> 32);
  return (low >> (shift & 31) & mask) != 0;
}

__device__ inline int GetIsotropicLOD(ulong mask, uint idx) {
  if (mask == 0) return 4;
  uint currHalf = idx < 32 ? uint(mask) : uint(mask >> 32);
  if ((currHalf >> (idx & 0x0Au) & 0x00330033u) == 0) return 2;
  return 1;
}

__device__ inline int popcnt64(ulong mask) {
  return __popcll(mask);
}

__device__ inline uint prefix_popcnt64(ulong mask, uint width) {
  uint lo = uint(mask);
  uint count = 0;
  if (width >= 32) {
    count = __popc(lo);
    lo = uint(mask >> 32);
  }
  uint m = 1u << (width & 31u);
  count += __popc(lo & (m - 1u));
  return count;
}

__device__ inline float3 ClipRayToAABB(float3 origin, float3 dir, float3 invDir,
                                        float3 boxMin, float3 boxMax) {
  float3 t0 = (boxMin - origin) * invDir;
  float3 t1 = (boxMax - origin) * invDir;
  float3 tmin = fminf(t0, t1);
  float3 tmax = fmaxf(t0, t1);
  float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
  float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);
  if (tNear <= tFar && tFar > 0) {
    return origin + dir * fmaxf(tNear, 0.0f);
  }
  return origin;
}

__device__ inline void AlignToCellBoundaries(int3 &ipos, float3 dir, int lod) {
  int cellMask = lod - 1;
  ipos.x = (dir.x < 0) ? (ipos.x & ~cellMask) : (ipos.x | cellMask);
  ipos.y = (dir.y < 0) ? (ipos.y & ~cellMask) : (ipos.y | cellMask);
  ipos.z = (dir.z < 0) ? (ipos.z & ~cellMask) : (ipos.z | cellMask);
}

__device__ inline uint ReadIndirection(SVO_TEXTURE_TYPE indirection, uint3 pos,
                                        int indW, int indH, int indD) {
  return indirection[pos.x + pos.y * indW + pos.z * indW * indH];
}

__device__ inline bool GetStepPos(int3 &ipos, float3 dir,
                                   SVO_TEXTURE_TYPE indirection,
                                   SECTOR_BUFFER_TYPE sectors, OCCUPANCY_TYPE occupancy,
                                   DATA_TYPE data, MASK_TYPE sectorMasks,
                                   int3 worldOrigin, uint8_t &outMatID,
                                   int indW, int indH, int indD) {
  if (ipos.y < 0 || ipos.y >= int(SIZEY)) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  int3 worldSector = make_int3(ipos.x >> 5, ipos.y >> 5, ipos.z >> 5);
  int3 relSector = worldSector - worldOrigin;

  if (relSector.x < 0 || relSector.x >= indW || relSector.y < 0 ||
      relSector.y >= indH || relSector.z < 0 || relSector.z >= indD) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  uint wx = (worldSector.x % indW + indW) % indW;
  uint wy = (worldSector.y % indH + indH) % indH;
  uint wz = (worldSector.z % indD + indD) % indD;
  uint3 sectorPos = make_uint3(wx, wy, wz);
  uint sectorIndex = ReadIndirection(indirection, sectorPos, indW, indH, indD);

  if (sectorIndex == SECTOR_HANDLE_EMPTY) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  SectorInfo sec = sectors[sectorIndex];
  uint64_t brickMask = sec.brickMask;

  uint3 localPos = make_uint3((uint)ipos.x & 31u, (uint)ipos.y & 31u, (uint)ipos.z & 31u);
  uint3 brickRel = (localPos >> 3) & 3;
  uint brickLinearIdx = GetLinearIndex4(brickRel);

  uint dirOctant = (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) + (dir.z >= 0 ? 4u : 0u);

  if (sec.flags == SECTOR_FLAG_LOD) {
    if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
      outMatID = MAT_STONE;
      return false;
    } else {
      ulong maskedBrick = brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
      int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
      AlignToCellBoundaries(ipos, dir, lod * 8);
      return true;
    }
  }

  if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
    uint packedBrickOffset = prefix_popcnt64(brickMask, brickLinearIdx);
    uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;

    uint3 subPos = (localPos >> 2) & 1;
    uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4);

    ulong voxMask = occupancy[occIndexBase + subIdx];

    uint3 vRel = localPos & 3;
    uint vIdx = GetLinearIndex4(vRel);

    if (BitTestHalf64(voxMask, vIdx, 1)) {
      if (data != nullptr) {
        uint dataIdx = (sec.baseBrickIndex + packedBrickOffset) * 512 + (subIdx * 64) + vIdx;
        outMatID = data[dataIdx];
      } else {
        outMatID = 1;
      }
      return false;
    }

    ulong maskedOcc = voxMask & RayMaskOptimizationLUT[vIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedOcc, vIdx);
    AlignToCellBoundaries(ipos, dir, lod);
    return true;
  } else {
    ulong maskedBrick = brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
    AlignToCellBoundaries(ipos, dir, lod * 8);
    return true;
  }
}

__device__ inline hitInfo trace(float3 rayPos, float3 rayDir,
                                 SVO_TEXTURE_TYPE indirection,
                                 SECTOR_BUFFER_TYPE sectors, OCCUPANCY_TYPE occupancy,
                                 DATA_TYPE data, MASK_TYPE sectorMasks,
                                 int3 worldOrigin, int indW, int indH, int indD,
                                 CHAR_DATA_TYPE charData = nullptr) {
  hitInfo hit;
  hit.hit = false;
  hit.normal = make_half3(0, 1, 0);
  hit.its = 0;

  float3 safeDir = rayDir;
  safeDir.x = (fabsf(safeDir.x) < 1e-8f) ? copysignf(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (fabsf(safeDir.y) < 1e-8f) ? copysignf(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (fabsf(safeDir.z) < 1e-8f) ? copysignf(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = make_float3(1.0f / safeDir.x, 1.0f / safeDir.y, 1.0f / safeDir.z);

  float3 worldMin = make_float3(worldOrigin.x * 32.0f, worldOrigin.y * 32.0f, worldOrigin.z * 32.0f);
  float3 worldMax = make_float3((worldOrigin.x + indW) * 32.0f,
                                (worldOrigin.y + indH) * 32.0f,
                                (worldOrigin.z + indD) * 32.0f);

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
  int3 voxelPos = make_int3((int)floorf(startPos.x), (int)floorf(startPos.y), (int)floorf(startPos.z));
  
  float closestCharT = 1e20f;
  float3 closestCharNormal = make_float3(0.0f, 0.0f, 0.0f);
  int closestCharMatID = 0;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          mat4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = mat4_mul_float3(invBBox, rayDir, false);
          float3 localBPos = mat4_mul_float3(invBBox, rayPos, true);
          
          float3 safeLocalBDir = localBDir;
          safeLocalBDir.x = (fabsf(safeLocalBDir.x) < 1e-8f) ? copysignf(1e-8f, safeLocalBDir.x) : safeLocalBDir.x;
          safeLocalBDir.y = (fabsf(safeLocalBDir.y) < 1e-8f) ? copysignf(1e-8f, safeLocalBDir.y) : safeLocalBDir.y;
          safeLocalBDir.z = (fabsf(safeLocalBDir.z) < 1e-8f) ? copysignf(1e-8f, safeLocalBDir.z) : safeLocalBDir.z;
          float3 invLocalBDir = make_float3(1.0f / safeLocalBDir.x, 1.0f / safeLocalBDir.y, 1.0f / safeLocalBDir.z);
          
          float3 bt0 = make_float3((-0.5f - localBPos.x) * invLocalBDir.x,
                                   (-0.5f - localBPos.y) * invLocalBDir.y,
                                   (-0.5f - localBPos.z) * invLocalBDir.z);
          float3 bt1 = make_float3(( 0.5f - localBPos.x) * invLocalBDir.x,
                                   ( 0.5f - localBPos.y) * invLocalBDir.y,
                                   ( 0.5f - localBPos.z) * invLocalBDir.z);
          float3 btmin = fminf(bt0, bt1);
          float3 btmax = fmaxf(bt0, bt1);
          float bNear = fmaxf(fmaxf(btmin.x, btmin.y), btmin.z);
          float bFar = fminf(fminf(btmax.x, btmax.y), btmax.z);
          
          if (bNear < 0.0f && bFar > 0.0f) continue;
          if (bNear > bFar || bFar < 0.0f || bNear >= closestCharT) continue;
          
          for (int p = 0; p < 6; ++p) {
              mat4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = mat4_mul_float3(invPart, rayDir, false);
              float3 localPos = mat4_mul_float3(invPart, rayPos, true);
              
              float3 safeLocalDir = localDir;
              safeLocalDir.x = (fabsf(safeLocalDir.x) < 1e-8f) ? copysignf(1e-8f, safeLocalDir.x) : safeLocalDir.x;
              safeLocalDir.y = (fabsf(safeLocalDir.y) < 1e-8f) ? copysignf(1e-8f, safeLocalDir.y) : safeLocalDir.y;
              safeLocalDir.z = (fabsf(safeLocalDir.z) < 1e-8f) ? copysignf(1e-8f, safeLocalDir.z) : safeLocalDir.z;
              float3 invLocalDir = make_float3(1.0f / safeLocalDir.x, 1.0f / safeLocalDir.y, 1.0f / safeLocalDir.z);
              
              float3 t0 = make_float3((-0.5f - localPos.x) * invLocalDir.x,
                                      (-0.5f - localPos.y) * invLocalDir.y,
                                      (-0.5f - localPos.z) * invLocalDir.z);
              float3 t1 = make_float3(( 0.5f - localPos.x) * invLocalDir.x,
                                      ( 0.5f - localPos.y) * invLocalDir.y,
                                      ( 0.5f - localPos.z) * invLocalDir.z);
              float3 tmin = fminf(t0, t1);
              float3 tmax = fmaxf(t0, t1);
              float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
              float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);
              
              if (tNear <= tFar && tFar > 0.0f && tNear < closestCharT) {
                  closestCharT = fmaxf(tNear, 0.0f);
                  closestCharMatID = 255;
                  
                  float3 hitPosLocal = make_float3(localPos.x + localDir.x * closestCharT,
                                                   localPos.y + localDir.y * closestCharT,
                                                   localPos.z + localDir.z * closestCharT);
                  float3 absHit = make_float3(fabsf(hitPosLocal.x), fabsf(hitPosLocal.y), fabsf(hitPosLocal.z));
                  float3 localNormal = make_float3(0.0f, 0.0f, 0.0f);
                  if (absHit.x > absHit.y && absHit.x > absHit.z)
                      localNormal = make_float3(copysignf(1.0f, hitPosLocal.x), 0.0f, 0.0f);
                  else if (absHit.y > absHit.z)
                      localNormal = make_float3(0.0f, copysignf(1.0f, hitPosLocal.y), 0.0f);
                  else
                      localNormal = make_float3(0.0f, 0.0f, copysignf(1.0f, hitPosLocal.z));
                      
                  mat4 transposed = mat4_transpose(invPart);
                  closestCharNormal = normalize(mat4_mul_float3(transposed, localNormal, false));
              }
          }
      }
      #endif
  }

  int maxIters = 512;
  for (int i = 0; i < maxIters; ++i) {
    hit.its++;

    if ((float)voxelPos.x < worldMin.x || (float)voxelPos.y < worldMin.y ||
        (float)voxelPos.z < worldMin.z || (float)voxelPos.x >= worldMax.x ||
        (float)voxelPos.y >= worldMax.y || (float)voxelPos.z >= worldMax.z) {
      break;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID,
                              indW, indH, indD);

    if (!stepped) {
      hit.hit = true;

      float3 cellMin = make_float3((float)voxelPos.x, (float)voxelPos.y, (float)voxelPos.z);
      float3 t0 = make_float3((cellMin.x - rayPos.x) * invDir.x,
                              (cellMin.y - rayPos.y) * invDir.y,
                              (cellMin.z - rayPos.z) * invDir.z);
      float3 t1 = make_float3((cellMin.x + 1.0f - rayPos.x) * invDir.x,
                              (cellMin.y + 1.0f - rayPos.y) * invDir.y,
                              (cellMin.z + 1.0f - rayPos.z) * invDir.z);
      float3 tmin_v = fminf(t0, t1);
      float tEntry = fmaxf(fmaxf(tmin_v.x, tmin_v.y), tmin_v.z);
      
      if (closestCharT < tEntry) {
          hit.pos = make_float3(rayPos.x + rayDir.x * closestCharT,
                               rayPos.y + rayDir.y * closestCharT,
                               rayPos.z + rayDir.z * closestCharT);
          hit.normal = make_half3(closestCharNormal.x, closestCharNormal.y, closestCharNormal.z);
          hit.matID = closestCharMatID;
          hit.uv = make_half2(0.0f, 0.0f);
          return hit;
      }

      hit.pos = make_float3(rayPos.x + rayDir.x * tEntry,
                           rayPos.y + rayDir.y * tEntry,
                           rayPos.z + rayDir.z * tEntry);

      float3 center = make_float3(cellMin.x + 0.5f, cellMin.y + 0.5f, cellMin.z + 0.5f);
      float3 d = make_float3(hit.pos.x - center.x, hit.pos.y - center.y, hit.pos.z - center.z);
      float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

      if (ad.x > ad.y && ad.x > ad.z)
        hit.normal = make_half3(copysignf(1.0f, d.x), 0, 0);
      else if (ad.y > ad.z)
        hit.normal = make_half3(0, copysignf(1.0f, d.y), 0);
      else
        hit.normal = make_half3(0, 0, copysignf(1.0f, d.z));

      float3 fpos = make_float3(floorf(hit.pos.x), floorf(hit.pos.y), floorf(hit.pos.z));
      float3 localPos = make_float3(hit.pos.x - fpos.x, hit.pos.y - fpos.y, hit.pos.z - fpos.z);
      if (fabsf(hit.normal.x) > 0.5f)
        hit.uv = make_half2(localPos.y, localPos.z);
      else if (fabsf(hit.normal.y) > 0.5f)
        hit.uv = make_half2(localPos.x, localPos.z);
      else
        hit.uv = make_half2(localPos.x, localPos.y);

      hit.matID = matID;
      return hit;
    }

    float3 nextBoundary;
    nextBoundary.x = (safeDir.x >= 0) ? ((float)voxelPos.x + 1.0f) : (float)voxelPos.x;
    nextBoundary.y = (safeDir.y >= 0) ? ((float)voxelPos.y + 1.0f) : (float)voxelPos.y;
    nextBoundary.z = (safeDir.z >= 0) ? ((float)voxelPos.z + 1.0f) : (float)voxelPos.z;

    float3 tMax;
    tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
    tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
    tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

    float tmin = fminf(fminf(tMax.x, tMax.y), tMax.z);

    int3 nextVoxelPos;
    nextVoxelPos.x = (int)floorf(startPos.x + tmin * safeDir.x);
    nextVoxelPos.y = (int)floorf(startPos.y + tmin * safeDir.y);
    nextVoxelPos.z = (int)floorf(startPos.z + tmin * safeDir.z);

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

  if (closestCharT < 1e20f) {
      hit.hit = true;
      hit.pos = make_float3(rayPos.x + rayDir.x * closestCharT,
                           rayPos.y + rayDir.y * closestCharT,
                           rayPos.z + rayDir.z * closestCharT);
      hit.normal = make_half3(closestCharNormal.x, closestCharNormal.y, closestCharNormal.z);
      hit.matID = closestCharMatID;
      hit.uv = make_half2(0.0f, 0.0f);
      return hit;
  }

  return hit;
}

__device__ inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                                    int maxIters, SVO_TEXTURE_TYPE indirection,
                                    SECTOR_BUFFER_TYPE sectors, OCCUPANCY_TYPE occupancy,
                                    DATA_TYPE data, MASK_TYPE sectorMasks,
                                    int3 worldOrigin, int indW, int indH, int indD,
                                    CHAR_DATA_TYPE charData = nullptr) {
  float3 safeDir = rayDir;
  safeDir.x = (fabsf(safeDir.x) < 1e-8f) ? copysignf(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (fabsf(safeDir.y) < 1e-8f) ? copysignf(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (fabsf(safeDir.z) < 1e-8f) ? copysignf(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = make_float3(1.0f / safeDir.x, 1.0f / safeDir.y, 1.0f / safeDir.z);

  float3 worldMin = make_float3(worldOrigin.x * 32.0f, worldOrigin.y * 32.0f, worldOrigin.z * 32.0f);
  float3 worldMax = make_float3((worldOrigin.x + indW) * 32.0f,
                                (worldOrigin.y + indH) * 32.0f,
                                (worldOrigin.z + indD) * 32.0f);

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
  float maxDistSq = maxDist * maxDist;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          mat4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = mat4_mul_float3(invBBox, rayDir, false);
          float3 localBPos = mat4_mul_float3(invBBox, rayPos, true);
          
          float3 safeLocalBDir = localBDir;
          safeLocalBDir.x = (fabsf(safeLocalBDir.x) < 1e-8f) ? copysignf(1e-8f, safeLocalBDir.x) : safeLocalBDir.x;
          safeLocalBDir.y = (fabsf(safeLocalBDir.y) < 1e-8f) ? copysignf(1e-8f, safeLocalBDir.y) : safeLocalBDir.y;
          safeLocalBDir.z = (fabsf(safeLocalBDir.z) < 1e-8f) ? copysignf(1e-8f, safeLocalBDir.z) : safeLocalBDir.z;
          float3 invLocalBDir = make_float3(1.0f / safeLocalBDir.x, 1.0f / safeLocalBDir.y, 1.0f / safeLocalBDir.z);
          
          float3 bt0 = make_float3((-0.5f - localBPos.x) * invLocalBDir.x,
                                   (-0.5f - localBPos.y) * invLocalBDir.y,
                                   (-0.5f - localBPos.z) * invLocalBDir.z);
          float3 bt1 = make_float3(( 0.5f - localBPos.x) * invLocalBDir.x,
                                   ( 0.5f - localBPos.y) * invLocalBDir.y,
                                   ( 0.5f - localBPos.z) * invLocalBDir.z);
          float3 btmin = fminf(bt0, bt1);
          float3 btmax = fmaxf(bt0, bt1);
          float bNear = fmaxf(fmaxf(btmin.x, btmin.y), btmin.z);
          float bFar = fminf(fminf(btmax.x, btmax.y), btmax.z);
          
          float bStartDist = fmaxf(0.0f, bNear);
          if (bNear > bFar || bFar < 0.0f || (bStartDist * bStartDist) > maxDistSq) continue;
          
          for (int p = 0; p < 6; ++p) {
              mat4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = mat4_mul_float3(invPart, rayDir, false);
              float3 localPos = mat4_mul_float3(invPart, rayPos, true);
              
              float3 safeLocalDir = localDir;
              safeLocalDir.x = (fabsf(safeLocalDir.x) < 1e-8f) ? copysignf(1e-8f, safeLocalDir.x) : safeLocalDir.x;
              safeLocalDir.y = (fabsf(safeLocalDir.y) < 1e-8f) ? copysignf(1e-8f, safeLocalDir.y) : safeLocalDir.y;
              safeLocalDir.z = (fabsf(safeLocalDir.z) < 1e-8f) ? copysignf(1e-8f, safeLocalDir.z) : safeLocalDir.z;
              float3 invLocalDir = make_float3(1.0f / safeLocalDir.x, 1.0f / safeLocalDir.y, 1.0f / safeLocalDir.z);
              
              float3 t0 = make_float3((-0.5f - localPos.x) * invLocalDir.x,
                                      (-0.5f - localPos.y) * invLocalDir.y,
                                      (-0.5f - localPos.z) * invLocalDir.z);
              float3 t1 = make_float3(( 0.5f - localPos.x) * invLocalDir.x,
                                      ( 0.5f - localPos.y) * invLocalDir.y,
                                      ( 0.5f - localPos.z) * invLocalDir.z);
              float3 tmin = fminf(t0, t1);
              float3 tmax = fmaxf(t0, t1);
              float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
              float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);
              
              float tStartDist = fmaxf(0.0f, tNear);
              if (tNear <= tFar && tFar > 0.0f && (tStartDist * tStartDist) < maxDistSq) {
                  return true;
              }
          }
      }
      #endif
  }

  int3 voxelPos = make_int3((int)floorf(startPos.x), (int)floorf(startPos.y), (int)floorf(startPos.z));

  for (int i = 0; i < maxIters; ++i) {
    if ((float)voxelPos.x < worldMin.x || (float)voxelPos.y < worldMin.y ||
        (float)voxelPos.z < worldMin.z || (float)voxelPos.x >= worldMax.x ||
        (float)voxelPos.y >= worldMax.y || (float)voxelPos.z >= worldMax.z) {
      return false;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID,
                              indW, indH, indD);

    if (!stepped) return true;

    float3 diff = make_float3((float)voxelPos.x + 0.5f - rayPos.x,
                              (float)voxelPos.y + 0.5f - rayPos.y,
                              (float)voxelPos.z + 0.5f - rayPos.z);
    if ((diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) > maxDistSq) return false;

    float3 nextBoundary;
    nextBoundary.x = (safeDir.x >= 0) ? ((float)voxelPos.x + 1.0f) : (float)voxelPos.x;
    nextBoundary.y = (safeDir.y >= 0) ? ((float)voxelPos.y + 1.0f) : (float)voxelPos.y;
    nextBoundary.z = (safeDir.z >= 0) ? ((float)voxelPos.z + 1.0f) : (float)voxelPos.z;

    float3 tMax;
    tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
    tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
    tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

    float tmin = fminf(fminf(tMax.x, tMax.y), tMax.z);

    int3 nextVoxelPos;
    nextVoxelPos.x = (int)floorf(startPos.x + tmin * safeDir.x);
    nextVoxelPos.y = (int)floorf(startPos.y + tmin * safeDir.y);
    nextVoxelPos.z = (int)floorf(startPos.z + tmin * safeDir.z);

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

#else
// ============================================================================
// C++ REFERENCE IMPLEMENTATION (for debugging/testing)
// ============================================================================
#include "renderer/ShaderTypes.h"
#include "renderer/shader_settings.h"
#include "renderer/hitInfo.h"
#include "tables.h"
#include <cmath>

using uint = unsigned int;
using uchar = unsigned char;
using ulong = uint64_t;

inline uint GetLinearIndex4(uint3 p) {
  return (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4);
}

inline bool BitTestHalf64(ulong value, uint shift, uint mask) {
  uint low = shift < 32 ? uint(value) : uint(value >> 32);
  return (low >> (shift & 31) & mask) != 0;
}

inline int GetIsotropicLOD(ulong mask, uint idx) {
  if (mask == 0) return 4;
  uint currHalf = idx < 32 ? uint(mask) : uint(mask >> 32);
  if ((currHalf >> (idx & 0x0Au) & 0x00330033u) == 0) return 2;
  return 1;
}

inline int popcnt64(ulong mask) {
  int count = 0;
  while (mask) {
    count += mask & 1;
    mask >>= 1;
  }
  return count;
}

inline uint prefix_popcnt64(ulong mask, uint width) {
  uint count = 0;
  for (uint i = 0; i < width; ++i) {
    count += (mask >> i) & 1;
  }
  return count;
}

inline float3 ClipRayToAABB(float3 origin, float3 dir, float3 invDir,
                            float3 boxMin, float3 boxMax) {
  float3 t0 = (boxMin - origin) * invDir;
  float3 t1 = (boxMax - origin) * invDir;
  float3 tmin = min(t0, t1);
  float3 tmax = max(t0, t1);
  float tNear = std::max(std::max(tmin.x, tmin.y), tmin.z);
  float tFar = std::min(std::min(tmax.x, tmax.y), tmax.z);
  if (tNear <= tFar && tFar > 0) {
    return origin + dir * std::max(tNear, 0.0f);
  }
  return origin;
}

inline void AlignToCellBoundaries(int3 &ipos, float3 dir, int lod) {
  int cellMask = lod - 1;
  ipos.x = (dir.x < 0) ? (ipos.x & ~cellMask) : (ipos.x | cellMask);
  ipos.y = (dir.y < 0) ? (ipos.y & ~cellMask) : (ipos.y | cellMask);
  ipos.z = (dir.z < 0) ? (ipos.z & ~cellMask) : (ipos.z | cellMask);
}

inline uint ReadIndirection(const uint32_t* indirection, uint3 pos, 
                            int indW, int indH, int indD) {
  (void)indH; (void)indD; // Unused but kept for API consistency
  return indirection[pos.x + pos.y * indW + pos.z * indW * indH];
}

inline bool GetStepPos(int3 &ipos, float3 dir,
                       const uint32_t* indirection,
                       const SectorInfo* sectors, const uint64_t* occupancy,
                       const uint8_t* data, const uint64_t* sectorMasks,
                       int3 worldOrigin, uint8_t &outMatID,
                       int indW, int indH, int indD) {
  (void)sectorMasks; // Unused in reference implementation
  
  if (ipos.y < 0 || ipos.y >= (int)SIZEY) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  int3 worldSector = make_int3(ipos.x >> 5, ipos.y >> 5, ipos.z >> 5);
  int3 relSector = worldSector - worldOrigin;

  if (relSector.x < 0 || relSector.x >= indW || relSector.y < 0 ||
      relSector.y >= indH || relSector.z < 0 || relSector.z >= indD) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  uint wx = (worldSector.x % indW + indW) % indW;
  uint wy = (worldSector.y % indH + indH) % indH;
  uint wz = (worldSector.z % indD + indD) % indD;
  uint3 sectorPos = make_uint3(wx, wy, wz);
  uint sectorIndex = ReadIndirection(indirection, sectorPos, indW, indH, indD);

  if (sectorIndex == SECTOR_HANDLE_EMPTY) {
    AlignToCellBoundaries(ipos, dir, 32);
    return true;
  }

  SectorInfo sec = sectors[sectorIndex];
  uint64_t brickMask = sec.brickMask;

  uint3 localPos = make_uint3((uint)ipos.x & 31u, (uint)ipos.y & 31u, (uint)ipos.z & 31u);
  uint3 brickRel = (localPos >> 3) & 3;
  uint brickLinearIdx = GetLinearIndex4(brickRel);

  uint dirOctant = (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) + (dir.z >= 0 ? 4u : 0u);

  if (sec.flags == SECTOR_FLAG_LOD) {
    if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
      outMatID = MAT_STONE;
      return false;
    } else {
      ulong maskedBrick = brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
      int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
      AlignToCellBoundaries(ipos, dir, lod * 8);
      return true;
    }
  }

  if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
    uint packedBrickOffset = prefix_popcnt64(brickMask, brickLinearIdx);
    uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;

    uint3 subPos = (localPos >> 2) & 1;
    uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4);

    ulong voxMask = occupancy[occIndexBase + subIdx];

    uint3 vRel = localPos & 3;
    uint vIdx = GetLinearIndex4(vRel);

    if (BitTestHalf64(voxMask, vIdx, 1)) {
      if (data != nullptr) {
        uint dataIdx = (sec.baseBrickIndex + packedBrickOffset) * 512 + (subIdx * 64) + vIdx;
        outMatID = data[dataIdx];
      } else {
        outMatID = 1;
      }
      return false;
    }

    ulong maskedOcc = voxMask & RayMaskOptimizationLUT[vIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedOcc, vIdx);
    AlignToCellBoundaries(ipos, dir, lod);
    return true;
  } else {
    ulong maskedBrick = brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
    int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
    AlignToCellBoundaries(ipos, dir, lod * 8);
    return true;
  }
}

inline hitInfo trace(float3 rayPos, float3 rayDir,
                     const uint32_t* indirection,
                     const SectorInfo* sectors, const uint64_t* occupancy,
                     const uint8_t* data, const uint64_t* sectorMasks,
                     int3 worldOrigin, int indW, int indH, int indD,
                     const CharacterGPUData* charData = nullptr) {
  (void)sectorMasks; // Unused in reference
  
  hitInfo hit;
  hit.hit = false;
  hit.normal = make_half3(0, 1, 0);
  hit.its = 0;

  float3 safeDir = rayDir;
  safeDir.x = (std::abs(safeDir.x) < 1e-8f) ? std::copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (std::abs(safeDir.y) < 1e-8f) ? std::copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (std::abs(safeDir.z) < 1e-8f) ? std::copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = make_float3(1.0f / safeDir.x, 1.0f / safeDir.y, 1.0f / safeDir.z);

  float3 worldMin = make_float3(worldOrigin.x * 32.0f, worldOrigin.y * 32.0f, worldOrigin.z * 32.0f);
  float3 worldMax = make_float3((worldOrigin.x + indW) * 32.0f,
                                (worldOrigin.y + indH) * 32.0f,
                                (worldOrigin.z + indD) * 32.0f);

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
  int3 voxelPos = make_int3((int)std::floor(startPos.x), (int)std::floor(startPos.y), (int)std::floor(startPos.z));
  
  float closestCharT = 1e20f;
  float3 closestCharNormal = make_float3(0.0f, 0.0f, 0.0f);
  int closestCharMatID = 0;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          mat4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = mat4_mul_float3(invBBox, rayDir, false);
          float3 localBPos = mat4_mul_float3(invBBox, rayPos, true);
          
          float3 safeLocalBDir = localBDir;
          safeLocalBDir.x = (std::abs(safeLocalBDir.x) < 1e-8f) ? std::copysign(1e-8f, safeLocalBDir.x) : safeLocalBDir.x;
          safeLocalBDir.y = (std::abs(safeLocalBDir.y) < 1e-8f) ? std::copysign(1e-8f, safeLocalBDir.y) : safeLocalBDir.y;
          safeLocalBDir.z = (std::abs(safeLocalBDir.z) < 1e-8f) ? std::copysign(1e-8f, safeLocalBDir.z) : safeLocalBDir.z;
          float3 invLocalBDir = make_float3(1.0f / safeLocalBDir.x, 1.0f / safeLocalBDir.y, 1.0f / safeLocalBDir.z);
          
          float3 bt0 = make_float3((-0.5f - localBPos.x) * invLocalBDir.x,
                                   (-0.5f - localBPos.y) * invLocalBDir.y,
                                   (-0.5f - localBPos.z) * invLocalBDir.z);
          float3 bt1 = make_float3(( 0.5f - localBPos.x) * invLocalBDir.x,
                                   ( 0.5f - localBPos.y) * invLocalBDir.y,
                                   ( 0.5f - localBPos.z) * invLocalBDir.z);
          float3 btmin = min(bt0, bt1);
          float3 btmax = max(bt0, bt1);
          float bNear = std::max(std::max(btmin.x, btmin.y), btmin.z);
          float bFar = std::min(std::min(btmax.x, btmax.y), btmax.z);
          
          if (bNear < 0.0f && bFar > 0.0f) continue;
          if (bNear > bFar || bFar < 0.0f || bNear >= closestCharT) continue;
          
          for (int p = 0; p < 6; ++p) {
              mat4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = mat4_mul_float3(invPart, rayDir, false);
              float3 localPos = mat4_mul_float3(invPart, rayPos, true);
              
              float3 safeLocalDir = localDir;
              safeLocalDir.x = (std::abs(safeLocalDir.x) < 1e-8f) ? std::copysign(1e-8f, safeLocalDir.x) : safeLocalDir.x;
              safeLocalDir.y = (std::abs(safeLocalDir.y) < 1e-8f) ? std::copysign(1e-8f, safeLocalDir.y) : safeLocalDir.y;
              safeLocalDir.z = (std::abs(safeLocalDir.z) < 1e-8f) ? std::copysign(1e-8f, safeLocalDir.z) : safeLocalDir.z;
              float3 invLocalDir = make_float3(1.0f / safeLocalDir.x, 1.0f / safeLocalDir.y, 1.0f / safeLocalDir.z);
              
              float3 t0 = make_float3((-0.5f - localPos.x) * invLocalDir.x,
                                      (-0.5f - localPos.y) * invLocalDir.y,
                                      (-0.5f - localPos.z) * invLocalDir.z);
              float3 t1 = make_float3(( 0.5f - localPos.x) * invLocalDir.x,
                                      ( 0.5f - localPos.y) * invLocalDir.y,
                                      ( 0.5f - localPos.z) * invLocalDir.z);
              float3 tmin = min(t0, t1);
              float3 tmax = max(t0, t1);
              float tNear = std::max(std::max(tmin.x, tmin.y), tmin.z);
              float tFar = std::min(std::min(tmax.x, tmax.y), tmax.z);
              
              if (tNear <= tFar && tFar > 0.0f && tNear < closestCharT) {
                  closestCharT = std::max(tNear, 0.0f);
                  closestCharMatID = 255;
                  
                  float3 hitPosLocal = make_float3(localPos.x + localDir.x * closestCharT,
                                                   localPos.y + localDir.y * closestCharT,
                                                   localPos.z + localDir.z * closestCharT);
                  float3 absHit = make_float3(std::abs(hitPosLocal.x), std::abs(hitPosLocal.y), std::abs(hitPosLocal.z));
                  float3 localNormal = make_float3(0.0f, 0.0f, 0.0f);
                  if (absHit.x > absHit.y && absHit.x > absHit.z)
                      localNormal = make_float3(std::copysign(1.0f, hitPosLocal.x), 0.0f, 0.0f);
                  else if (absHit.y > absHit.z)
                      localNormal = make_float3(0.0f, std::copysign(1.0f, hitPosLocal.y), 0.0f);
                  else
                      localNormal = make_float3(0.0f, 0.0f, std::copysign(1.0f, hitPosLocal.z));
                      
                  mat4 transposed = mat4_transpose(invPart);
                  closestCharNormal = normalize(mat4_mul_float3(transposed, localNormal, false));
              }
          }
      }
      #endif
  }

  int maxIters = 512;
  for (int i = 0; i < maxIters; ++i) {
    hit.its++;

    if ((float)voxelPos.x < worldMin.x || (float)voxelPos.y < worldMin.y ||
        (float)voxelPos.z < worldMin.z || (float)voxelPos.x >= worldMax.x ||
        (float)voxelPos.y >= worldMax.y || (float)voxelPos.z >= worldMax.z) {
      break;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID,
                              indW, indH, indD);

    if (!stepped) {
      hit.hit = true;

      float3 cellMin = make_float3((float)voxelPos.x, (float)voxelPos.y, (float)voxelPos.z);
      float3 t0 = make_float3((cellMin.x - rayPos.x) * invDir.x,
                              (cellMin.y - rayPos.y) * invDir.y,
                              (cellMin.z - rayPos.z) * invDir.z);
      float3 t1 = make_float3((cellMin.x + 1.0f - rayPos.x) * invDir.x,
                              (cellMin.y + 1.0f - rayPos.y) * invDir.y,
                              (cellMin.z + 1.0f - rayPos.z) * invDir.z);
      float3 tmin_v = min(t0, t1);
      float tEntry = std::max(std::max(tmin_v.x, tmin_v.y), tmin_v.z);
      
      if (closestCharT < tEntry) {
          hit.pos = make_float3(rayPos.x + rayDir.x * closestCharT,
                               rayPos.y + rayDir.y * closestCharT,
                               rayPos.z + rayDir.z * closestCharT);
          hit.normal = make_half3(closestCharNormal.x, closestCharNormal.y, closestCharNormal.z);
          hit.matID = closestCharMatID;
          hit.uv = make_half2(0.0f, 0.0f);
          return hit;
      }

      hit.pos = make_float3(rayPos.x + rayDir.x * tEntry,
                           rayPos.y + rayDir.y * tEntry,
                           rayPos.z + rayDir.z * tEntry);

      float3 center = make_float3(cellMin.x + 0.5f, cellMin.y + 0.5f, cellMin.z + 0.5f);
      float3 d = make_float3(hit.pos.x - center.x, hit.pos.y - center.y, hit.pos.z - center.z);
      float3 ad = make_float3(std::abs(d.x), std::abs(d.y), std::abs(d.z));

      if (ad.x > ad.y && ad.x > ad.z)
        hit.normal = make_half3(std::copysign(1.0f, d.x), 0, 0);
      else if (ad.y > ad.z)
        hit.normal = make_half3(0, std::copysign(1.0f, d.y), 0);
      else
        hit.normal = make_half3(0, 0, std::copysign(1.0f, d.z));

      float3 fpos = make_float3(std::floor(hit.pos.x), std::floor(hit.pos.y), std::floor(hit.pos.z));
      float3 localPos = make_float3(hit.pos.x - fpos.x, hit.pos.y - fpos.y, hit.pos.z - fpos.z);
      if (std::abs(hit.normal.x) > 0.5f)
        hit.uv = make_half2(localPos.y, localPos.z);
      else if (std::abs(hit.normal.y) > 0.5f)
        hit.uv = make_half2(localPos.x, localPos.z);
      else
        hit.uv = make_half2(localPos.x, localPos.y);

      hit.matID = matID;
      return hit;
    }

    float3 nextBoundary;
    nextBoundary.x = (safeDir.x >= 0) ? ((float)voxelPos.x + 1.0f) : (float)voxelPos.x;
    nextBoundary.y = (safeDir.y >= 0) ? ((float)voxelPos.y + 1.0f) : (float)voxelPos.y;
    nextBoundary.z = (safeDir.z >= 0) ? ((float)voxelPos.z + 1.0f) : (float)voxelPos.z;

    float3 tMax;
    tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
    tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
    tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

    float tmin = std::min(std::min(tMax.x, tMax.y), tMax.z);

    int3 nextVoxelPos;
    nextVoxelPos.x = (int)std::floor(startPos.x + tmin * safeDir.x);
    nextVoxelPos.y = (int)std::floor(startPos.y + tmin * safeDir.y);
    nextVoxelPos.z = (int)std::floor(startPos.z + tmin * safeDir.z);

    nextVoxelPos.x = (safeDir.x >= 0) ? std::max(nextVoxelPos.x, originalVoxelPos.x) : std::min(nextVoxelPos.x, originalVoxelPos.x);
    nextVoxelPos.y = (safeDir.y >= 0) ? std::max(nextVoxelPos.y, originalVoxelPos.y) : std::min(nextVoxelPos.y, originalVoxelPos.y);
    nextVoxelPos.z = (safeDir.z >= 0) ? std::max(nextVoxelPos.z, originalVoxelPos.z) : std::min(nextVoxelPos.z, originalVoxelPos.z);

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
      hit.pos = make_float3(rayPos.x + rayDir.x * closestCharT,
                           rayPos.y + rayDir.y * closestCharT,
                           rayPos.z + rayDir.z * closestCharT);
      hit.normal = make_half3(closestCharNormal.x, closestCharNormal.y, closestCharNormal.z);
      hit.matID = closestCharMatID;
      hit.uv = make_half2(0.0f, 0.0f);
      return hit;
  }

  return hit;
}

inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                        int maxIters, const uint32_t* indirection,
                        const SectorInfo* sectors, const uint64_t* occupancy,
                        const uint8_t* data, const uint64_t* sectorMasks,
                        int3 worldOrigin, int indW, int indH, int indD,
                        const CharacterGPUData* charData = nullptr) {
  (void)sectorMasks;
  
  float3 safeDir = rayDir;
  safeDir.x = (std::abs(safeDir.x) < 1e-8f) ? std::copysign(1e-8f, safeDir.x) : safeDir.x;
  safeDir.y = (std::abs(safeDir.y) < 1e-8f) ? std::copysign(1e-8f, safeDir.y) : safeDir.y;
  safeDir.z = (std::abs(safeDir.z) < 1e-8f) ? std::copysign(1e-8f, safeDir.z) : safeDir.z;
  float3 invDir = make_float3(1.0f / safeDir.x, 1.0f / safeDir.y, 1.0f / safeDir.z);

  float3 worldMin = make_float3(worldOrigin.x * 32.0f, worldOrigin.y * 32.0f, worldOrigin.z * 32.0f);
  float3 worldMax = make_float3((worldOrigin.x + indW) * 32.0f,
                                (worldOrigin.y + indH) * 32.0f,
                                (worldOrigin.z + indD) * 32.0f);

  float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
  float maxDistSq = maxDist * maxDist;
  
  if (charData != nullptr) {
      #if CHARACTER_MODELS
      for (int i = 0; i < charData->numCharacters; ++i) {
          mat4 invBBox = charData->invBoundingBoxes[i];
          float3 localBDir = mat4_mul_float3(invBBox, rayDir, false);
          float3 localBPos = mat4_mul_float3(invBBox, rayPos, true);
          
          float3 safeLocalBDir = localBDir;
          safeLocalBDir.x = (std::abs(safeLocalBDir.x) < 1e-8f) ? std::copysign(1e-8f, safeLocalBDir.x) : safeLocalBDir.x;
          safeLocalBDir.y = (std::abs(safeLocalBDir.y) < 1e-8f) ? std::copysign(1e-8f, safeLocalBDir.y) : safeLocalBDir.y;
          safeLocalBDir.z = (std::abs(safeLocalBDir.z) < 1e-8f) ? std::copysign(1e-8f, safeLocalBDir.z) : safeLocalBDir.z;
          float3 invLocalBDir = make_float3(1.0f / safeLocalBDir.x, 1.0f / safeLocalBDir.y, 1.0f / safeLocalBDir.z);
          
          float3 bt0 = make_float3((-0.5f - localBPos.x) * invLocalBDir.x,
                                   (-0.5f - localBPos.y) * invLocalBDir.y,
                                   (-0.5f - localBPos.z) * invLocalBDir.z);
          float3 bt1 = make_float3(( 0.5f - localBPos.x) * invLocalBDir.x,
                                   ( 0.5f - localBPos.y) * invLocalBDir.y,
                                   ( 0.5f - localBPos.z) * invLocalBDir.z);
          float3 btmin = min(bt0, bt1);
          float3 btmax = max(bt0, bt1);
          float bNear = std::max(std::max(btmin.x, btmin.y), btmin.z);
          float bFar = std::min(std::min(btmax.x, btmax.y), btmax.z);
          
          float bStartDist = std::max(0.0f, bNear);
          if (bNear > bFar || bFar < 0.0f || (bStartDist * bStartDist) > maxDistSq) continue;
          
          for (int p = 0; p < 6; ++p) {
              mat4 invPart = charData->invBodyParts[i * 6 + p];
              float3 localDir = mat4_mul_float3(invPart, rayDir, false);
              float3 localPos = mat4_mul_float3(invPart, rayPos, true);
              
              float3 safeLocalDir = localDir;
              safeLocalDir.x = (std::abs(safeLocalDir.x) < 1e-8f) ? std::copysign(1e-8f, safeLocalDir.x) : safeLocalDir.x;
              safeLocalDir.y = (std::abs(safeLocalDir.y) < 1e-8f) ? std::copysign(1e-8f, safeLocalDir.y) : safeLocalDir.y;
              safeLocalDir.z = (std::abs(safeLocalDir.z) < 1e-8f) ? std::copysign(1e-8f, safeLocalDir.z) : safeLocalDir.z;
              float3 invLocalDir = make_float3(1.0f / safeLocalDir.x, 1.0f / safeLocalDir.y, 1.0f / safeLocalDir.z);
              
              float3 t0 = make_float3((-0.5f - localPos.x) * invLocalDir.x,
                                      (-0.5f - localPos.y) * invLocalDir.y,
                                      (-0.5f - localPos.z) * invLocalDir.z);
              float3 t1 = make_float3(( 0.5f - localPos.x) * invLocalDir.x,
                                      ( 0.5f - localPos.y) * invLocalDir.y,
                                      ( 0.5f - localPos.z) * invLocalDir.z);
              float3 tmin = min(t0, t1);
              float3 tmax = max(t0, t1);
              float tNear = std::max(std::max(tmin.x, tmin.y), tmin.z);
              float tFar = std::min(std::min(tmax.x, tmax.y), tmax.z);
              
              float tStartDist = std::max(0.0f, tNear);
              if (tNear <= tFar && tFar > 0.0f && (tStartDist * tStartDist) < maxDistSq) {
                  return true;
              }
          }
      }
      #endif
  }

  int3 voxelPos = make_int3((int)std::floor(startPos.x), (int)std::floor(startPos.y), (int)std::floor(startPos.z));

  for (int i = 0; i < maxIters; ++i) {
    if ((float)voxelPos.x < worldMin.x || (float)voxelPos.y < worldMin.y ||
        (float)voxelPos.z < worldMin.z || (float)voxelPos.x >= worldMax.x ||
        (float)voxelPos.y >= worldMax.y || (float)voxelPos.z >= worldMax.z) {
      return false;
    }

    uint8_t matID = 0;
    int3 originalVoxelPos = voxelPos;

    bool stepped = GetStepPos(voxelPos, safeDir, indirection, sectors,
                              occupancy, data, sectorMasks, worldOrigin, matID,
                              indW, indH, indD);

    if (!stepped) return true;

    float3 diff = make_float3((float)voxelPos.x + 0.5f - rayPos.x,
                              (float)voxelPos.y + 0.5f - rayPos.y,
                              (float)voxelPos.z + 0.5f - rayPos.z);
    if ((diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) > maxDistSq) return false;

    float3 nextBoundary;
    nextBoundary.x = (safeDir.x >= 0) ? ((float)voxelPos.x + 1.0f) : (float)voxelPos.x;
    nextBoundary.y = (safeDir.y >= 0) ? ((float)voxelPos.y + 1.0f) : (float)voxelPos.y;
    nextBoundary.z = (safeDir.z >= 0) ? ((float)voxelPos.z + 1.0f) : (float)voxelPos.z;

    float3 tMax;
    tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
    tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
    tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

    float tmin = std::min(std::min(tMax.x, tMax.y), tMax.z);

    int3 nextVoxelPos;
    nextVoxelPos.x = (int)std::floor(startPos.x + tmin * safeDir.x);
    nextVoxelPos.y = (int)std::floor(startPos.y + tmin * safeDir.y);
    nextVoxelPos.z = (int)std::floor(startPos.z + tmin * safeDir.z);

    nextVoxelPos.x = (safeDir.x >= 0) ? std::max(nextVoxelPos.x, originalVoxelPos.x) : std::min(nextVoxelPos.x, originalVoxelPos.x);
    nextVoxelPos.y = (safeDir.y >= 0) ? std::max(nextVoxelPos.y, originalVoxelPos.y) : std::min(nextVoxelPos.y, originalVoxelPos.y);
    nextVoxelPos.z = (safeDir.z >= 0) ? std::max(nextVoxelPos.z, originalVoxelPos.z) : std::min(nextVoxelPos.z, originalVoxelPos.z);

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

#endif
