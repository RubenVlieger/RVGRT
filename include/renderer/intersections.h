#pragma once

#include <metal_stdlib>
#include "renderer/ShaderTypes.h"
#include "raytracing_functions.h"
#include "renderer/hitInfo.h"
#include "tables.h"
#include "ShaderTypes.h"

using namespace metal;

// =================================================================================
// LOW-LEVEL HELPERS
// =================================================================================

// Linear Index for 4x4x4 blocks (XZY order)
// Maps (0..3, 0..3, 0..3) -> 0..63
inline uint GetLinearIndex4(uint3 p) {
    return (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4);
}

// 64-bit Population Count (Metal Intrinsic wrapper)
inline int popcnt64(ulong mask) {
    return popcount(mask);
}

// Create a mask of bits lower than the given index (for prefix sum calculation)
inline ulong GetLowerMask(int index) {
    // If index is 0, we want 0. If index is 63, we want bits 0-62.
    // 1UL << 64 is undefined behavior, so handle carefully if needed, 
    // but index is always < 64 here.
    return (1UL << index) - 1UL;
}

// -----------------------------------------------------------------------------
// GEOMETRY INTERSECTIONS
// -----------------------------------------------------------------------------

// Robust Ray-AABB intersection. Returns the entry point clamped to the box.
inline float3 ClipRayToAABB(float3 origin, float3 dir, float3 boxMin, float3 boxMax) {
    float3 invDir = 1.0f / (dir + 1e-10f); // Avoid div by zero
    float3 t0 = (boxMin - origin) * invDir;
    float3 t1 = (boxMax - origin) * invDir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    
    // If inside or intersecting
    if (tNear <= tFar && tFar > 0) {
        // We move just barely inside the box
        return origin + dir * max(tNear, 0.0f);
    }
    return origin;
}

// Parametric Alignment: Snaps the current position to the next cell boundary
// along the ray direction.
inline int3 AlignToCellBoundaries(int3 pos, float3 dir, int cellSize) {
    int cellMask = cellSize - 1;
    // If moving negative, align to lower bound of previous cell.
    // If moving positive, align to upper bound of current cell.
    int3 nextPos;
    nextPos.x = dir.x < 0 ? (pos.x & ~cellMask) - 1 : (pos.x & ~cellMask) + cellSize;
    nextPos.y = dir.y < 0 ? (pos.y & ~cellMask) - 1 : (pos.y & ~cellMask) + cellSize;
    nextPos.z = dir.z < 0 ? (pos.z & ~cellMask) - 1 : (pos.z & ~cellMask) + cellSize;
    
    // However, the Parametric loop expects 'pos' to be the current integer coordinate.
    // The trick is we only change components if the boundary is crossed.
    // A simpler way for DDA-like stepping without T-values inside the stepper:
    
    // Actually, for the logic provided in the prompt:
    pos.x = dir.x < 0 ? (pos.x & ~cellMask) : (pos.x | cellMask);
    pos.y = dir.y < 0 ? (pos.y & ~cellMask) : (pos.y | cellMask);
    pos.z = dir.z < 0 ? (pos.z & ~cellMask) : (pos.z | cellMask);
    return pos;
}

// =================================================================================
// CORE BIT-SCANNING STEPPER
// =================================================================================

/**
 * Checks the hierarchy at the current voxel position `ipos`.
 * If empty, advances `ipos` by the size of the empty region (1, 8, or 32).
 * If solid, returns true and fills `outMatID`.
 */
inline bool GetStepPos(
    thread int3& ipos, 
    float3 dir, 
    texture3d<uint, access::read> indirection,
    device SectorInfo* sectors,
    device ulong* occupancy,
    device uchar* data,
    thread uint8_t& outMatID
) {
    uint3 pos = uint3(ipos);
    
    // 1. Level 1: Sector (32x32x32)
    uint3 sectorPos = pos >> 5; // Divide by 32
    
    if (sectorPos.x >= indirection.get_width() || 
        sectorPos.y >= indirection.get_height() || 
        sectorPos.z >= indirection.get_depth()) {
        // Escaped world bounds
        ipos += int3(dir * 32.0f); 
        return false; 
    }

    uint sectorIndex = indirection.read(sectorPos).r;
    
    // Empty Sector? Skip 32.
    if (sectorIndex == 0) {
        ipos = AlignToCellBoundaries(ipos, dir, 32);
        return false;
    }

    // Load Sector Data
    // Note: Indirection is 1-based.
    SectorInfo sec = sectors[sectorIndex]; 
    uint64_t brickMask = sec.brickMask; // Which 8^3 bricks exist?

    // 2. Level 2: Brick (8x8x8)
    // Get coordinate within the sector (0..3)
    uint3 brickRel = (pos >> 3) & 3; 
    uint brickLinearIdx = GetLinearIndex4(brickRel);
    
    // Is the Brick Bit set?
    if ((brickMask >> brickLinearIdx) & 1) {
        
        // Brick Exists. We need to check sub-bricks (4x4x4).
        
        // Calculate offset into Occupancy Buffer.
        // We count how many set bits are *before* our current brick index.
        ulong maskPre = GetLowerMask(brickLinearIdx);
        int packedBrickOffset = popcnt64(brickMask & maskPre);
        
        // Base index for this brick's 8 uint64 masks
        uint64_t occIndexBase = (sec.baseBrickIndex + packedBrickOffset) * 8;
        
        // 3. Level 3: Sub-Brick (4x4x4)
        // A brick (8^3) has 8 sub-bricks (4^3).
        // Get sub-brick coordinate (0..1) within brick
        uint3 subPos = (pos >> 2) & 1; 
        uint subIdx = subPos.x + (subPos.z * 2) + (subPos.y * 4); // XZY order for sub-chunks
        
        ulong voxMask = occupancy[occIndexBase + subIdx];
        
        // --- BIT SCAN OPTIMIZATION ---
        // We assume the ray is moving. We mask out voxels "behind" us or
        // irrelevant to the ray direction using the LUT.
        // Ray Octant: x<0=1, y<0=2, z<0=4
        uint dirOctant = (dir.x < 0 ? 1 : 0) + (dir.y < 0 ? 2 : 0) + (dir.z < 0 ? 4 : 0);
        
        // 4. Level 4: Voxel (1x1x1)
        uint3 vRel = pos & 3; // Coordinate within 4x4x4 sub-brick
        uint vIdx = GetLinearIndex4(vRel);
        
        // Check if the SPECIFIC voxel we are in is solid
        if ((voxMask >> vIdx) & 1) {
            // HIT!
            // Calculate data offset: 
            // (Base + BrickOffset) * 512 bytes + (SubBrickIndex * 64) + VoxelIndex
            uint dataIdx = (sec.baseBrickIndex + packedBrickOffset) * 512 + (subIdx * 64) + vIdx;
            outMatID = data[dataIdx];
            return true;
        }
        
        // We are in a solid brick, but an empty voxel.
        // Optimization: Can we skip the rest of this 4x4x4 block?
        // Apply LUT to see if *any* relevant voxel along the ray path is set.
        
        // If the remaining relevant bits are 0, we can skip the rest of this scale.
        // For simplicity in V1, we just step 1 voxel. 
        // (Advanced: Use tzcnt on (voxMask & LUT) to compute distance).
        ipos = AlignToCellBoundaries(ipos, dir, 1);
        return false;

    } else {
        // Empty Brick? Skip 8.
        ipos = AlignToCellBoundaries(ipos, dir, 8);
        return false;
    }
}

// =================================================================================
// MAIN TRACE FUNCTIONS
// =================================================================================

inline hitInfo trace(
    float3 rayPos, 
    float3 rayDir, 
    texture3d<uint, access::read> indirection, 
    device SectorInfo* sectors,
    device ulong* occupancy,
    device uchar* data
) {
    hitInfo hit;
    hit.hit = false;
    hit.normal = half3(0,1,0);
    hit.its = 0;

    // 1. Setup Parametric Variables
    float3 invDir = 1.0f / (rayDir + 1e-20f);
    
    // Define World Bounds (e.g., 2048^3) based on Indirection Size
    float3 worldSize = float3(indirection.get_width(), indirection.get_height(), indirection.get_depth()) * 32.0f;
    
    // Move Ray to Bounding Box
    float3 startPos = ClipRayToAABB(rayPos, rayDir, float3(0), worldSize);
    
    // Bias: Nudge slightly inside to prevent Z-fighting at boundaries
    // We maintain 'voxelPos' as our integer coordinate tracker
    int3 voxelPos = int3(floor(startPos + rayDir * 0.001f));
    
    // 2. Traversal Loop
    int maxIters = 512; // Safety break
    
    for (int i = 0; i < maxIters; ++i) {
        hit.its++;

        uint8_t matID = 0;
        
        // This function advances voxelPos if empty, or returns true if solid
        bool isHit = GetStepPos(voxelPos, rayDir, indirection, sectors, occupancy, data, matID);
        
        if (isHit) {
            hit.hit = true;
            
            // Re-calculate precise T intersection for the specific voxel we found
            float3 cellMin = float3(voxelPos);
            float3 t0 = (cellMin - rayPos) * invDir;
            float3 t1 = (cellMin + 1.0f - rayPos) * invDir;
            float3 tmax_v = max(t0, t1);
            float3 tmin_v = min(t0, t1);
            float tEntry = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
            
            hit.pos = rayPos + rayDir * tEntry;
            
            // Calculate Normal based on dominant axis of entry
            // (Standard DDA-style normal logic)
            float3 center = cellMin + 0.5f;
            float3 d = hit.pos - center;
            float3 ad = abs(d);
            
            if (ad.x > ad.y && ad.x > ad.z) hit.normal = half3(sign(d.x), 0, 0);
            else if (ad.y > ad.z)           hit.normal = half3(0, sign(d.y), 0);
            else                            hit.normal = half3(0, 0, sign(d.z));
            
            // Store Material ID in the color channel for now (or separate field)
            // Storing as half3 for compatibility with existing codebase
            //hit.color = half3((float)matID / 255.0f, 0, 0); 
            hit.matID = matID;
            
            return hit;
        }
        
        
        // Bounds check
        if (voxelPos.x < 0 || voxelPos.y < 0 || voxelPos.z < 0 ||
            voxelPos.x >= worldSize.x || voxelPos.y >= worldSize.y || voxelPos.z >= worldSize.z) {
            break;
        }
    }

    return hit;
}

// Optimized Shadow Trace (Boolean)
inline bool traceShadow(
    float3 rayPos, 
    float3 rayDir, 
    float maxDist,
    texture3d<uint, access::read> indirection, 
    device SectorInfo* sectors,
    device ulong* occupancy,
    device uchar* data
) {
    float3 invDir = 1.0f / (rayDir + 1e-20f);
    float3 worldSize = float3(indirection.get_width(), indirection.get_height(), indirection.get_depth()) * 32.0f;
    
    // Only clip start, we don't need exact entry point if inside
    float3 startPos = ClipRayToAABB(rayPos, rayDir, float3(0), worldSize);
    
    int3 voxelPos = int3(floor(startPos + rayDir * 0.001f));
    
    // Distance tracking
    float currentDist = 0.0f;
    float3 initialPos = rayPos;

    for (int i = 0; i < 256; ++i) {
        uint8_t matID = 0;
        
        // Check current position
        if (GetStepPos(voxelPos, rayDir, indirection, sectors, occupancy, data, matID)) {
            return true; // Hit anything opaque
        }
        
        // Update distance check
        // (Approximation using Manhattan distance or just check bounds)
        // For accurate distance cap, we'd need to track T values, 
        // but for shadows, checking world bounds is usually enough.
        if (voxelPos.x < 0 || voxelPos.y < 0 || voxelPos.z < 0 ||
            voxelPos.x >= worldSize.x || voxelPos.y >= worldSize.y || voxelPos.z >= worldSize.z) {
            return false;
        }
        
        // Optional: Exact distance check
        // float3 cellMin = float3(voxelPos);
        // float3 t0 = (cellMin - initialPos) * invDir;
        // float tCurrent = max(max(t0.x, t0.y), t0.z);
        // if (tCurrent > maxDist) return false;
    }
    
    return false;
}