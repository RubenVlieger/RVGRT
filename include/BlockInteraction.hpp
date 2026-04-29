#pragma once

#include <cstdint>

// ============================================================================
// BlockInteraction.hpp
//
// Data types for voxel block placement and removal (Phase 2).
//
// ARCHITECTURE
// ─────────────
// Block edits flow through two parallel paths:
//
// 1. Overlay Map (g_blockEdits in VoxelQuery.hpp):
//    A hash map from world voxel coordinate → matID that overrides
//    the procedural Evaluate() function. This is checked first by
//    IsVoxelSolid(), so collision and DDA raycasting immediately
//    reflect all edits.
//
// 2. SVO Data (MaterialMap::RemoveVoxel/PlaceVoxel):
//    Direct modification of the GPU-side occupancy bits and material
//    data bytes in the Metal brick pool shared-memory buffers.
//    This makes edits visible to the path tracer.
//
// When a block is edited, BOTH paths are updated simultaneously.
// The overlay map is the source of truth for which voxels have been
// modified; the SVO data mirrors those changes into the renderer.
//
// PHASE 4 UPGRADE: Network Sync
// ──────────────────────────────
// BlockEdit structs will be serialized to/from the server via
// NetworkClient::SendBlockEdit() and BlockEditCallback. The server
// maintains an authoritative list and broadcasts incremental changes.
// On connect, a full sync (BlockSyncMessage) applies all remote edits.
// The /reset command triggers BlockResetMessage which clears both
// the overlay map and SVO data.
// ============================================================================

struct RaycastResult {
    bool hit = false;
    int32_t voxelX = 0;
    int32_t voxelY = 0;
    int32_t voxelZ = 0;
    int32_t adjacentX = 0;
    int32_t adjacentY = 0;
    int32_t adjacentZ = 0;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 0.0f;
    uint8_t matID = 0;
    float distance = 0.0f;
};

struct BlockEdit {
    int32_t x;
    int32_t y;
    int32_t z;
    uint8_t matID;
};

enum class BlockAction { Remove, Place };