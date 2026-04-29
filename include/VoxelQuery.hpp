#pragma once

// ============================================================================
// VoxelQuery.hpp
//
// CPU-side voxel solidity queries and raycasting for the procedural world.
//
// HOW IT WORKS
// ─────────────
// The world is defined procedurally by TerrainGeneration::Evaluate(x, y, z).
// A voxel at grid coordinate (ix, iy, iz) has its center at
// (ix + 0.5f, iy + 0.5f, iz + 0.5f). Evaluate() returns >0 for solid, ≤0 for air.
//
// IsVoxelSolid(ix, iy, iz) first checks a block-edit overlay map, then falls
// back to Evaluate(). This immediately makes all collision and raycasting
// consistent with block modifications.
//
// RaycastDDA(origin, direction, maxDist) traces a ray through the voxel grid
// using a DDA (Digital Differential Analyzer) stepping algorithm, calling
// IsVoxelSolid() at each step. It returns a RaycastResult with the hit voxel
// coordinate, the adjacent air voxel (for block placement), and the face normal.
//
// PHASE 2 UPGRADE: SVO Raycaster
// ───────────────────────────────
// The DDA approach above is sufficient for short-range block targeting (8 blocks)
// but walks one voxel at a time. For longer rays or performance-critical paths,
// a full SVO raycaster can be added that wraps the C++ reference trace() from
// intersections.h:
//
//   MaterialMap::Raycast(pos, dir, maxDist) would:
//   1. Convert rayPos/rayDir from glm to cumath types
//   2. Call the C++ trace() with _indirectionCPU, _sectorInfoCPU,
//      and CPU mirrors of occupancy/data buffers
//   3. Convert the hitInfo result to RaycastResult
//
//   GPU readback strategy:
//   - After GenerateDetailBatch() completes, issue a blocking GPU→CPU blit
//     to copy occupancy/data buffers
//   - Store as std::vector<uint64_t> _occupancyCPU and
//     std::vector<uint8_t> _dataCPU mirrors in MaterialMap
//   - Mark dirty on block edit; re-upload after modification
//
//   The trace() function already computes hit position, normal, matID, and UV —
//   it just needs CPU-accessible buffer pointers. The DDA approach below is
//   simpler and sufficient for the 8-block reach of block interaction.
// ============================================================================

#include "TerrainGeneration.h"
#include "BlockInteraction.hpp"
#include <cmath>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

// ─── Block Edit Overlay Map ──────────────────────────────────────────────────
//
// Global hash map storing runtime block modifications that override the
// procedural terrain. Key = world voxel coordinate, Value = matID (0 = air).
// Checked by IsVoxelSolid() BEFORE calling Evaluate().
//
extern std::unordered_map<glm::ivec3, uint8_t> g_blockEdits;

inline void SetBlockEdit(int x, int y, int z, uint8_t matID) {
    g_blockEdits[{x, y, z}] = matID;
}

inline void RemoveBlockEdit(int x, int y, int z) {
    g_blockEdits.erase({x, y, z});
}

inline void ClearAllBlockEdits() {
    g_blockEdits.clear();
}

// ─── IsVoxelSolid ─────────────────────────────────────────────────────────────
//
// Returns true if the voxel at integer grid position (x, y, z) is solid.
//
// Evaluation order:
//   1. Check g_blockEdits overlay map (runtime modifications take priority)
//   2. Fall back to Evaluate() (procedural terrain density)
//
// Evaluation point: voxel center + tiny offset to avoid sitting exactly on
// a boundary where Evaluate() could return exactly 0.
inline bool IsVoxelSolid(int x, int y, int z) {
    auto it = g_blockEdits.find({x, y, z});
    if (it != g_blockEdits.end())
        return it->second != 0;
    return Evaluate(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f) > 0.0f;
}

// ─── RaycastDDA ───────────────────────────────────────────────────────────────
//
// Traces a ray through the voxel grid using DDA (Digital Differential Analyzer)
// stepping. At each voxel boundary crossing, it calls IsVoxelSolid() to check
// if the voxel is solid. This naturally integrates with the block-edit overlay
// map, making raycasting immediately consistent with all block modifications.
//
// Parameters:
//   origin    - Ray origin in world space (player eye position)
//   direction - Ray direction (will be normalized internally)
//   maxDist   - Maximum ray distance in blocks (typical: 8.0)
//
// Returns:
//   RaycastResult with hit position, adjacent air position (for placement),
//   face normal, material ID, and distance. If no hit, .hit = false.
//
inline RaycastResult RaycastDDA(glm::vec3 origin, glm::vec3 direction, float maxDist) {
    RaycastResult result;
    result.hit = false;

    float len = glm::length(direction);
    if (len < 1e-8f)
        return result;
    glm::vec3 dir = direction / len;

    int ix = (int)std::floor(origin.x);
    int iy = (int)std::floor(origin.y);
    int iz = (int)std::floor(origin.z);

    int stepX = dir.x >= 0.0f ? 1 : -1;
    int stepY = dir.y >= 0.0f ? 1 : -1;
    int stepZ = dir.z >= 0.0f ? 1 : -1;

    float tDeltaX = (dir.x != 0.0f) ? std::abs(1.0f / dir.x) : 1e30f;
    float tDeltaY = (dir.y != 0.0f) ? std::abs(1.0f / dir.y) : 1e30f;
    float tDeltaZ = (dir.z != 0.0f) ? std::abs(1.0f / dir.z) : 1e30f;

    float tMaxX, tMaxY, tMaxZ;
    if (dir.x > 0.0f) {
        tMaxX = (std::floor(origin.x) + 1.0f - origin.x) / dir.x;
    } else if (dir.x < 0.0f) {
        tMaxX = (origin.x - std::floor(origin.x)) / (-dir.x);
    } else {
        tMaxX = 1e30f;
    }
    if (dir.y > 0.0f) {
        tMaxY = (std::floor(origin.y) + 1.0f - origin.y) / dir.y;
    } else if (dir.y < 0.0f) {
        tMaxY = (origin.y - std::floor(origin.y)) / (-dir.y);
    } else {
        tMaxY = 1e30f;
    }
    if (dir.z > 0.0f) {
        tMaxZ = (std::floor(origin.z) + 1.0f - origin.z) / dir.z;
    } else if (dir.z < 0.0f) {
        tMaxZ = (origin.z - std::floor(origin.z)) / (-dir.z);
    } else {
        tMaxZ = 1e30f;
    }

    int lastAxis = -1; // 0=X, 1=Y, 2=Z
    float t = 0.0f;
    int maxSteps = (int)(maxDist * 3.0f) + 10;

    for (int step = 0; step < maxSteps; ++step) {
        if (t > maxDist)
            break;

        if (IsVoxelSolid(ix, iy, iz)) {
            result.hit = true;
            result.voxelX = ix;
            result.voxelY = iy;
            result.voxelZ = iz;

            if (lastAxis == 0) {
                result.distance = tMaxX - tDeltaX;
                result.normalX = (float)(-stepX);
                result.normalY = 0.0f;
                result.normalZ = 0.0f;
            } else if (lastAxis == 1) {
                result.distance = tMaxY - tDeltaY;
                result.normalX = 0.0f;
                result.normalY = (float)(-stepY);
                result.normalZ = 0.0f;
            } else if (lastAxis == 2) {
                result.distance = tMaxZ - tDeltaZ;
                result.normalX = 0.0f;
                result.normalY = 0.0f;
                result.normalZ = (float)(-stepZ);
            } else {
                result.distance = 0.0f;
                result.normalX = 0.0f;
                result.normalY = 1.0f;
                result.normalZ = 0.0f;
            }

            result.adjacentX = ix + (int32_t)result.normalX;
            result.adjacentY = iy + (int32_t)result.normalY;
            result.adjacentZ = iz + (int32_t)result.normalZ;

            auto it = g_blockEdits.find({ix, iy, iz});
            result.matID = (it != g_blockEdits.end()) ? it->second : 1;
            return result;
        }

        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t = tMaxX;
                tMaxX += tDeltaX;
                ix += stepX;
                lastAxis = 0;
            } else {
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                iz += stepZ;
                lastAxis = 2;
            }
        } else {
            if (tMaxY < tMaxZ) {
                t = tMaxY;
                tMaxY += tDeltaY;
                iy += stepY;
                lastAxis = 1;
            } else {
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                iz += stepZ;
                lastAxis = 2;
            }
        }
    }

    return result;
}