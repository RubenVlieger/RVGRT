#pragma once

// Core ray-voxel intersection for Slang
// Unified implementation for all targets
// Accesses global resources directly to work around Slang compiler limitations
// with passing resources alongside scalars to user functions.


#include "Settings.slang"
#include "Tables.slang"

struct hitInfo {
    float3 pos;
    half3 normal;
    half2 uv;
    bool hit;
    int its;
    uint matID;
};

[ForceInline]
inline uint GetLinearIndex4(uint3 p) {
    return (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4);
}
[ForceInline]
inline bool BitTestHalf64(uint64_t value, uint shift, uint mask) {
    uint low = shift < 32 ? uint(value) : uint(value >> 32);
    return (low >> (shift & 31) & mask) != 0;
}
[ForceInline]
inline int GetIsotropicLOD(uint64_t mask, uint idx) {
    if (mask == 0)
        return 4;
    uint currHalf = idx < 32 ? uint(mask) : uint(mask >> 32);
    if ((currHalf >> (idx & 0x0Au) & 0x00330033u) == 0)
        return 2;
    return 1;
}
[ForceInline]
inline int popcnt64(uint64_t mask) { return countbits(mask); }

[ForceInline]
inline uint prefix_popcnt64(uint64_t mask, uint width) {
    uint lo = uint(mask);
    uint count = 0;
    if (width >= 32) {
        count = countbits(lo);
        lo = uint(mask >> 32);
    }
    uint m = 1u << (width & 31u);
    count += countbits(lo & (m - 1u));
    return count;
}

[ForceInline]
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

[ForceInline]
inline void AlignToCellBoundaries(inout int3 ipos, float3 dir, int lod) {
    int cellMask = lod - 1;
    ipos.x = (dir.x < 0) ? (ipos.x & ~cellMask) : (ipos.x | cellMask);
    ipos.y = (dir.y < 0) ? (ipos.y & ~cellMask) : (ipos.y | cellMask);
    ipos.z = (dir.z < 0) ? (ipos.z & ~cellMask) : (ipos.z | cellMask);
}

[ForceInline]
inline uint ReadIndirection(uint3 pos) {
    return indirection[pos];
}

[ForceInline]
inline bool GetStepPos(inout int3 ipos, float3 dir,
                        int3 worldOrigin,
                        inout uint outMatID,
                        int indW, int indH, int indD,
                        bool lodOnly = false) {

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
    uint sectorIndex = ReadIndirection(sectorPos);

    if (sectorIndex == SECTOR_HANDLE_EMPTY) {
        AlignToCellBoundaries(ipos, dir, 32);
        return true;
    }

    SectorInfo sec = sectorBuffer[sectorIndex];
    uint64_t brickMask = sec.brickMask;

    uint3 localPos =
        uint3(uint(ipos.x) & 31u, uint(ipos.y) & 31u, uint(ipos.z) & 31u);
    uint3 brickRel = (localPos >> 3) & 3;
    uint brickLinearIdx = GetLinearIndex4(brickRel);

    uint dirOctant =
        (dir.x >= 0 ? 1u : 0u) + (dir.y >= 0 ? 2u : 0u) + (dir.z >= 0 ? 4u : 0u);

    if (sec.flags == SECTOR_FLAG_LOD || lodOnly) {
        if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
            outMatID = MAT_STONE;
            return false;
        } else {
            uint64_t maskedBrick =
                brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
            int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
            AlignToCellBoundaries(ipos, dir, lod * 8);
            return true;
        }
    }

    if (BitTestHalf64(brickMask, brickLinearIdx, 1)) {
        if (lodOnly) {
            outMatID = MAT_STONE;
            return false;
        }
        uint packedBrickOffset = prefix_popcnt64(brickMask, brickLinearIdx);
        uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;

        uint3 subPos = (localPos >> 2) & 1;
        uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4);

        uint64_t voxMask = occupancyBuffer[occIndexBase + subIdx];

        uint3 vRel = localPos & 3;
        uint vIdx = GetLinearIndex4(vRel);

        if (BitTestHalf64(voxMask, vIdx, 1)) {
            uint dataIdx = (sec.baseBrickIndex + packedBrickOffset) * 512 +
                           (subIdx * 64) + vIdx;
            outMatID = dataBuffer[dataIdx];
            return false;
        }

        uint64_t maskedOcc = voxMask & RayMaskOptimizationLUT[vIdx + dirOctant * 64];
        int lod = GetIsotropicLOD(maskedOcc, vIdx);
        AlignToCellBoundaries(ipos, dir, lod);
        return true;
    } else {
        uint64_t maskedBrick =
            brickMask & RayMaskOptimizationLUT[brickLinearIdx + dirOctant * 64];
        int lod = GetIsotropicLOD(maskedBrick, brickLinearIdx);
        AlignToCellBoundaries(ipos, dir, lod * 8);
        return true;
    }
}

[ForceInline]
inline hitInfo traceVoxels(float3 rayPos, float3 rayDir,
                            int3 worldOrigin, int indW, int indH, int indD,
                            float closestCharT, float3 closestCharNormal, int closestCharMatID,
                            int maxIters, bool lodOnly) {
    hitInfo hit;
    hit.hit = false;
    hit.normal = half3(0, 1, 0);
    hit.its = 0;

    float3 safeDir = rayDir;
    safeDir.x = (abs(safeDir.x) < 1e-8f) ? (safeDir.x < 0 ? -1e-8f : 1e-8f) : safeDir.x;
    safeDir.y = (abs(safeDir.y) < 1e-8f) ? (safeDir.y < 0 ? -1e-8f : 1e-8f) : safeDir.y;
    safeDir.z = (abs(safeDir.z) < 1e-8f) ? (safeDir.z < 0 ? -1e-8f : 1e-8f) : safeDir.z;
    float3 invDir = 1.0f / safeDir;

    float3 worldMin = float3(worldOrigin) * 32.0f;
    float3 worldMax =
        float3(worldOrigin.x + indW, worldOrigin.y + indH, worldOrigin.z + indD) *
        32.0f;

    float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
    int3 voxelPos = int3(floor(startPos));

    for (int i = 0; i < maxIters; ++i) {
        hit.its++;

        if (float(voxelPos.x) < worldMin.x || float(voxelPos.y) < worldMin.y ||
            float(voxelPos.z) < worldMin.z || float(voxelPos.x) >= worldMax.x ||
            float(voxelPos.y) >= worldMax.y || float(voxelPos.z) >= worldMax.z) {
            break;
        }

        uint matID = 0;
        int3 originalVoxelPos = voxelPos;

        bool stepped =
            GetStepPos(voxelPos, safeDir, worldOrigin, matID, indW, indH, indD, lodOnly);

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
                hit.normal = half3(d.x < 0 ? half(-1.0h) : half(1.0h), half(0.0h), half(0.0h));
            else if (ad.y > ad.z)
                hit.normal = half3(half(0.0h), d.y < 0 ? half(-1.0h) : half(1.0h), half(0.0h));
            else
                hit.normal = half3(half(0.0h), half(0.0h), d.z < 0 ? half(-1.0h) : half(1.0h));

            float3 fpos = floor(hit.pos);
            float3 localPos = hit.pos - fpos;
            if (abs(hit.normal.x) > 0.5h)
                hit.uv = half2(half(localPos.y), half(localPos.z));
            else if (abs(hit.normal.y) > 0.5h)
                hit.uv = half2(half(localPos.x), half(localPos.z));
            else
                hit.uv = half2(half(localPos.x), half(localPos.y));

            hit.matID = matID;
            return hit;
        }

        float3 nextBoundary;
        nextBoundary.x =
            (safeDir.x >= 0) ? (float(voxelPos.x) + 1.0f) : float(voxelPos.x);
        nextBoundary.y =
            (safeDir.y >= 0) ? (float(voxelPos.y) + 1.0f) : float(voxelPos.y);
        nextBoundary.z =
            (safeDir.z >= 0) ? (float(voxelPos.z) + 1.0f) : float(voxelPos.z);

        float3 tMax;
        tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
        tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
        tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

        float tmin = min(min(tMax.x, tMax.y), tMax.z);

        int3 nextVoxelPos;
        nextVoxelPos.x = int(floor(startPos.x + tmin * safeDir.x));
        nextVoxelPos.y = int(floor(startPos.y + tmin * safeDir.y));
        nextVoxelPos.z = int(floor(startPos.z + tmin * safeDir.z));

        nextVoxelPos.x = (safeDir.x >= 0) ? max(nextVoxelPos.x, originalVoxelPos.x)
                                          : min(nextVoxelPos.x, originalVoxelPos.x);
        nextVoxelPos.y = (safeDir.y >= 0) ? max(nextVoxelPos.y, originalVoxelPos.y)
                                          : min(nextVoxelPos.y, originalVoxelPos.y);
        nextVoxelPos.z = (safeDir.z >= 0) ? max(nextVoxelPos.z, originalVoxelPos.z)
                                          : min(nextVoxelPos.z, originalVoxelPos.z);

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

[ForceInline]
inline hitInfo trace(float3 rayPos, float3 rayDir,
              int3 worldOrigin, int indW, int indH, int indD,
              int maxIters = 512, bool lodOnly = false) {
    float closestCharT = 1e20f;
    float3 closestCharNormal = float3(0.0f);
    int closestCharMatID = 0;

    #if CHARACTER_MODELS
    for (int i = 0; i < charData.numCharacters; ++i) {
        float3 charCenter = charData.characterCenters[i].xyz;
        float3 toChar = charCenter - rayPos;
        float distSq = dot(toChar, toChar);
        if (distSq > CHARACTER_MAX_TRACE_DIST_SQ)
            continue;
        if (dot(toChar, rayDir) < 0.0f)
            continue;
        float4x4 invBBox = charData.invBoundingBoxes[i];
        float3 localBDir = mul(invBBox, float4(rayDir, 0.0f)).xyz;
        float3 localBPos = mul(invBBox, float4(rayPos, 1.0f)).xyz;

        float3 safeLocalBDir = localBDir;
        safeLocalBDir.x = (abs(safeLocalBDir.x) < 1e-8f) ? (safeLocalBDir.x < 0 ? -1e-8f : 1e-8f) : safeLocalBDir.x;
        safeLocalBDir.y = (abs(safeLocalBDir.y) < 1e-8f) ? (safeLocalBDir.y < 0 ? -1e-8f : 1e-8f) : safeLocalBDir.y;
        safeLocalBDir.z = (abs(safeLocalBDir.z) < 1e-8f) ? (safeLocalBDir.z < 0 ? -1e-8f : 1e-8f) : safeLocalBDir.z;
        float3 invLocalBDir = 1.0f / safeLocalBDir;

        float3 bt0 = (-0.5f - localBPos) * invLocalBDir;
        float3 bt1 = (0.5f - localBPos) * invLocalBDir;
        float3 btmin = min(bt0, bt1);
        float3 btmax = max(bt0, bt1);
        float bNear = max(max(btmin.x, btmin.y), btmin.z);
        float bFar = min(min(btmax.x, btmax.y), btmax.z);

        if (bNear < 0.0f && bFar > 0.0f)
            continue;
        if (bNear > bFar || bFar < 0.0f || bNear >= closestCharT)
            continue;

        for (int p = 0; p < 6; ++p) {
            float4x4 invPart = charData.invBodyParts[i * 6 + p];
            float3 localDir = mul(invPart, float4(rayDir, 0.0f)).xyz;
            float3 localPos = mul(invPart, float4(rayPos, 1.0f)).xyz;

            float3 safeLocalDir = localDir;
            safeLocalDir.x = (abs(safeLocalDir.x) < 1e-8f) ? (safeLocalDir.x < 0 ? -1e-8f : 1e-8f) : safeLocalDir.x;
            safeLocalDir.y = (abs(safeLocalDir.y) < 1e-8f) ? (safeLocalDir.y < 0 ? -1e-8f : 1e-8f) : safeLocalDir.y;
            safeLocalDir.z = (abs(safeLocalDir.z) < 1e-8f) ? (safeLocalDir.z < 0 ? -1e-8f : 1e-8f) : safeLocalDir.z;
            float3 invLocalDir = 1.0f / safeLocalDir;

            float3 t0 = (-0.5f - localPos) * invLocalDir;
            float3 t1 = (0.5f - localPos) * invLocalDir;
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
                    localNormal = float3(hitPosLocal.x < 0 ? -1 : 1, 0.0f, 0.0f);
                else if (absHit.y > absHit.z)
                    localNormal = float3(0.0f, hitPosLocal.y < 0 ? -1 : 1, 0.0f);
                else
                    localNormal = float3(0.0f, 0.0f, hitPosLocal.z < 0 ? -1 : 1);

                closestCharNormal = normalize(mul(transpose(invPart), float4(localNormal, 0.0f)).xyz);
            }
        }
    }
    #endif

    return traceVoxels(rayPos, rayDir, worldOrigin, indW, indH, indD,
                       closestCharT, closestCharNormal, closestCharMatID,
                       maxIters, lodOnly);
}

[ForceInline]
inline bool traceShadow(float3 rayPos, float3 rayDir, float maxDist,
                 int maxIters, int3 worldOrigin,
                 int indW, int indH, int indD) {
    float3 safeDir = rayDir;
    safeDir.x = (abs(safeDir.x) < 1e-8f) ? (safeDir.x < 0 ? -1e-8f : 1e-8f) : safeDir.x;
    safeDir.y = (abs(safeDir.y) < 1e-8f) ? (safeDir.y < 0 ? -1e-8f : 1e-8f) : safeDir.y;
    safeDir.z = (abs(safeDir.z) < 1e-8f) ? (safeDir.z < 0 ? -1e-8f : 1e-8f) : safeDir.z;
    float3 invDir = 1.0f / safeDir;

    float3 worldMin = float3(worldOrigin) * 32.0f;
    float3 worldMax =
        float3(worldOrigin.x + indW, worldOrigin.y + indH, worldOrigin.z + indD) *
        32.0f;

    float3 startPos = ClipRayToAABB(rayPos, safeDir, invDir, worldMin, worldMax);
    float maxDistSq = maxDist * maxDist;

    #if CHARACTER_MODELS
    for (int i = 0; i < charData.numCharacters; ++i) {
        float3 charCenter = charData.characterCenters[i].xyz;
        float3 toChar = charCenter - rayPos;
        float distSq = dot(toChar, toChar);
        if (distSq > CHARACTER_MAX_TRACE_DIST_SQ)
            continue;
        float4x4 invBBox = charData.invBoundingBoxes[i];
        float3 localBDir = mul(invBBox, float4(rayDir, 0.0f)).xyz;
        float3 localBPos = mul(invBBox, float4(rayPos, 1.0f)).xyz;

        float3 safeLocalBDir = localBDir;
        safeLocalBDir.x = (abs(safeLocalBDir.x) < 1e-8f) ? (safeLocalBDir.x < 0 ? -1e-8f : 1e-8f) : safeLocalBDir.x;
        safeLocalBDir.y = (abs(safeLocalBDir.y) < 1e-8f) ? (safeLocalBDir.y < 0 ? -1e-8f : 1e-8f) : safeLocalBDir.y;
        safeLocalBDir.z = (abs(safeLocalBDir.z) < 1e-8f) ? (safeLocalBDir.z < 0 ? -1e-8f : 1e-8f) : safeLocalBDir.z;
        float3 invLocalBDir = 1.0f / safeLocalBDir;

        float3 bt0 = (-0.5f - localBPos) * invLocalBDir;
        float3 bt1 = (0.5f - localBPos) * invLocalBDir;
        float3 btmin = min(bt0, bt1);
        float3 btmax = max(bt0, bt1);
        float bNear = max(max(btmin.x, btmin.y), btmin.z);
        float bFar = min(min(btmax.x, btmax.y), btmax.z);

        float bStartDist = max(0.0f, bNear);
        if (bNear > bFar || bFar < 0.0f || (bStartDist * bStartDist) > maxDistSq)
            continue;

        for (int p = 0; p < 6; ++p) {
            float4x4 invPart = charData.invBodyParts[i * 6 + p];
            float3 localDir = mul(invPart, float4(rayDir, 0.0f)).xyz;
            float3 localPos = mul(invPart, float4(rayPos, 1.0f)).xyz;

            float3 safeLocalDir = localDir;
            safeLocalDir.x = (abs(safeLocalDir.x) < 1e-8f) ? (safeLocalDir.x < 0 ? -1e-8f : 1e-8f) : safeLocalDir.x;
            safeLocalDir.y = (abs(safeLocalDir.y) < 1e-8f) ? (safeLocalDir.y < 0 ? -1e-8f : 1e-8f) : safeLocalDir.y;
            safeLocalDir.z = (abs(safeLocalDir.z) < 1e-8f) ? (safeLocalDir.z < 0 ? -1e-8f : 1e-8f) : safeLocalDir.z;
            float3 invLocalDir = 1.0f / safeLocalDir;

            float3 t0 = (-0.5f - localPos) * invLocalDir;
            float3 t1 = (0.5f - localPos) * invLocalDir;
            float3 tmin = min(t0, t1);
            float3 tmax = max(t0, t1);
            float tNear = max(max(tmin.x, tmin.y), tmin.z);
            float tFar = min(min(tmax.x, tmax.y), tmax.z);

            float tStartDist = max(0.0f, tNear);
            if (tNear <= tFar && tFar > 0.0f &&
                (tStartDist * tStartDist) < maxDistSq) {
                return true;
            }
        }
    }
    #endif

    int3 voxelPos = int3(floor(startPos));

    for (int i = 0; i < maxIters; ++i) {
        if (float(voxelPos.x) < worldMin.x || float(voxelPos.y) < worldMin.y ||
            float(voxelPos.z) < worldMin.z || float(voxelPos.x) >= worldMax.x ||
            float(voxelPos.y) >= worldMax.y || float(voxelPos.z) >= worldMax.z) {
            return false;
        }

        uint matID = 0;
        int3 originalVoxelPos = voxelPos;

        bool stepped =
            GetStepPos(voxelPos, safeDir, worldOrigin, matID, indW, indH, indD);

        if (!stepped)
            return true;

        float3 diff = float3(voxelPos) + 0.5f - rayPos;
        if (dot(diff, diff) > maxDistSq)
            return false;

        float3 nextBoundary;
        nextBoundary.x =
            (safeDir.x >= 0) ? (float(voxelPos.x) + 1.0f) : float(voxelPos.x);
        nextBoundary.y =
            (safeDir.y >= 0) ? (float(voxelPos.y) + 1.0f) : float(voxelPos.y);
        nextBoundary.z =
            (safeDir.z >= 0) ? (float(voxelPos.z) + 1.0f) : float(voxelPos.z);

        float3 tMax;
        tMax.x = (nextBoundary.x - startPos.x) * invDir.x;
        tMax.y = (nextBoundary.y - startPos.y) * invDir.y;
        tMax.z = (nextBoundary.z - startPos.z) * invDir.z;

        float tmin = min(min(tMax.x, tMax.y), tMax.z);

        int3 nextVoxelPos;
        nextVoxelPos.x = int(floor(startPos.x + tmin * safeDir.x));
        nextVoxelPos.y = int(floor(startPos.y + tmin * safeDir.y));
        nextVoxelPos.z = int(floor(startPos.z + tmin * safeDir.z));

        nextVoxelPos.x = (safeDir.x >= 0) ? max(nextVoxelPos.x, originalVoxelPos.x)
                                          : min(nextVoxelPos.x, originalVoxelPos.x);
        nextVoxelPos.y = (safeDir.y >= 0) ? max(nextVoxelPos.y, originalVoxelPos.y)
                                          : min(nextVoxelPos.y, originalVoxelPos.y);
        nextVoxelPos.z = (safeDir.z >= 0) ? max(nextVoxelPos.z, originalVoxelPos.z)
                                          : min(nextVoxelPos.z, originalVoxelPos.z);

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
