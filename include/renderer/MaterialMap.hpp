#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void *id;
#endif

#include "renderer/BrickPool.hpp"
#include "renderer/ShaderTypes.h"
#include <cstdint>
#include <mutex>
#include <simd/simd.h>
#include <vector>

// Forward declaration of implementation
namespace MaterialMapImpl {
    struct SectorState;
    struct AsyncResult;
}

/**
 * MaterialMap — Manages the voxel world with streaming and LOD.
 *
 * Owns:
 *   - Indirection 3D texture (toroidal, wraps around camera)
 *   - SectorInfo buffer (per-sector metadata)
 *   - BrickPool (occupancy + data buffers)
 *   - Super-sector mask buffer
 *
 * Call UpdateStreaming() each frame with the camera position.
 * Returns true if new sectors were loaded (caller should reset temporal
 * accumulation).
 */
class MaterialMap {
public:
  MaterialMap();
  ~MaterialMap();

  /// Initial world generation around spawn point.
  void GenerateDynamic();

  /// Per-frame streaming update. Returns true if any sectors changed.
  bool UpdateStreaming(simd_float3 cameraPos);

  // Getters for rendering bindings
  id GetIndirectionTexture();
  id GetSectorBuffer();
  id GetOccupancyBuffer();
  id GetDataBuffer();
  id GetSectorMaskBuffer();

  /// Get the world origin (world-space coordinate of indirection cell (0,0,0))
  simd_int3 GetWorldOrigin() const { return _worldOrigin; }

private:
  id _device;

  // --- GPU Resources ---
  // L1: 3D Texture (R32Uint). Value = sector handle (index into SectorBuffer)
  id _indirectionTexture;

  // L2: Buffer of SectorInfo structs
  id _sectorBuffer;

  // L3+L4: Brick data (owned by BrickPool)
  MetalBrickPool _brickPool;

  // Super-sector masks: one uint64_t per 4x4x4 group of sectors
  id _sectorMaskBuffer;

  // --- Compute Pipelines ---
  id _psoAnalyze;          // Analyze sectors (determine brick activity)
  id _psoFill;             // Fill brick data (occupancy + materials)
  id _psoAnalyzeLOD;       // LOD analysis (16 samples per brick for solid/air)
  id _psoAnalyzeStreaming; // Async streaming analysis of sectors

  // --- Async Compute Results ---
  struct AsyncResult {
    SectorWorkItem item;
    uint64_t brickMask;
  };
  std::mutex _asyncResultsMutex;
  std::vector<AsyncResult> _asyncResults;

  // --- Streaming State ---
  simd_int3 _worldOrigin; // World-space coordinate of indirection cell (0,0,0)
  simd_int3 _lastCameraSector; // Camera's sector position at last update
  bool _firstUpdate;

  // Indirection dimensions
  int _indW, _indH, _indD;

  // Per-sector tracking (indexed by wrapped indirection linear index)
  struct SectorState {
    int32_t worldX, worldY, worldZ; // World-space sector this slot represents
    uint32_t sectorHandle;  // Handle in sector buffer (1-based, 0 = unused)
    uint32_t brickPoolBase; // Base index in brick pool
    uint32_t brickCount;    // Number of allocated bricks
    bool isLoaded;
    bool isLOD;       // True = LOD (brickMask only, no data)
    bool isAnalyzing; // True if GPU analysis is currently pending
  };
  std::vector<SectorState> _sectorStates;

  // Sector buffer management
  uint32_t _nextSectorHandle;
  std::vector<uint32_t> _freeSectorHandles;
  std::vector<SectorInfo>
      _sectorInfoCPU; // CPU mirror of sector buffer (for updates)
  std::vector<uint32_t> _indirectionCPU; // CPU mirror of indirection texture

  // Command queue for async generation
  id _commandQueue;

  // --- Internal Methods ---

  /// Convert world sector position to wrapped indirection coordinates
  void WorldToWrapped(int wx, int wy, int wz, int &ix, int &iy, int &iz) const;

  /// Get linear index from wrapped coordinates
  int WrappedToLinear(int ix, int iy, int iz) const;

  /// Load a sector at world position. If isLOD, only generate brickMask.
  void LoadSector(int wx, int wy, int wz, bool isLOD,
                  std::vector<BrickWorkItem> &workList);

  /// Unload the sector at wrapped position
  void UnloadSector(int ix, int iy, int iz);

  /// Allocate a sector handle
  uint32_t AllocSectorHandle();

  /// Free a sector handle
  void FreeSectorHandle(uint32_t handle);

  /// Upload indirection texture for given wrapped coords
  void UploadIndirectionCell(int ix, int iy, int iz, uint32_t value);

  /// Upload a single SectorInfo to the GPU buffer
  void UploadSectorInfo(uint32_t handle, const SectorInfo &info);

  /// Generate full-detail bricks for a batch of sectors
  void GenerateDetailBatch(const std::vector<SectorWorkItem> &sectors);

  /// Generate LOD brickMasks for a batch of sectors
  void GenerateLODBatch(const std::vector<SectorWorkItem> &sectors);

  /// Rebuild super-sector masks for affected regions
  void RebuildSectorMasks();
};
