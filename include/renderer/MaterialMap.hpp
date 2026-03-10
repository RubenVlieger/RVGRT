#pragma once

#include "renderer/BrickPool.hpp"
#include "renderer/ShaderTypes.h"
#include <cstdint>
#include <mutex>
#include <vector>

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
 *
 * Platform-specific implementations live in MaterialMap.mm (Metal)
 * and CudaMaterialMap.cu (CUDA).
 */
class MaterialMap {
public:
  MaterialMap();
  ~MaterialMap();

  /// Initial world generation around spawn point.
  void GenerateDynamic();

  /// Per-frame streaming update. Returns true if any sectors changed.
  bool UpdateStreaming(simd_float3 cameraPos);

  // Getters for rendering bindings (return platform-specific handles as void*)
  // On Metal: these return id<MTLTexture> / id<MTLBuffer>
  // On CUDA:  these return device pointers (uint32_t*, SectorInfo*, etc.)
  void* GetIndirectionTexture();
  void* GetSectorBuffer();
  void* GetOccupancyBuffer();
  void* GetDataBuffer();
  void* GetSectorMaskBuffer();

  /// Get the world origin (world-space coordinate of indirection cell (0,0,0))
  simd_int3 GetWorldOrigin() const { return _worldOrigin; }

  /// Get indirection dimensions (needed by CUDA path to index flat buffer)
  int GetIndW() const { return _indW; }
  int GetIndH() const { return _indH; }
  int GetIndD() const { return _indD; }

private:
  void* _device; // Platform-specific device handle

  // --- GPU Resources ---
  // L1: 3D Texture / flat buffer (R32Uint). Value = sector handle
  void* _indirectionTexture;

  // L2: Buffer of SectorInfo structs
  void* _sectorBuffer;

  // L3+L4: Brick data (owned by BrickPool)
  BrickPool _brickPool;

  // Super-sector masks: one uint64_t per 4x4x4 group of sectors
  void* _sectorMaskBuffer;

  // --- Compute Pipelines (Metal PSOs or CUDA function pointers) ---
  void* _psoAnalyze;
  void* _psoFill;
  void* _psoAnalyzeLOD;
  void* _psoAnalyzeStreaming;

public:
  // --- Async Compute Results (public so CUDA callbacks can access) ---
  struct AsyncResult {
    SectorWorkItem item;
    uint64_t brickMask;
  };
  std::mutex _asyncResultsMutex;
  std::vector<AsyncResult> _asyncResults;

private:

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

  // Command queue for async generation (Metal) / CUDA stream (CUDA)
  void* _commandQueue;

  // --- Internal Methods ---
  void WorldToWrapped(int wx, int wy, int wz, int &ix, int &iy, int &iz) const;
  int WrappedToLinear(int ix, int iy, int iz) const;
  void LoadSector(int wx, int wy, int wz, bool isLOD,
                  std::vector<BrickWorkItem> &workList);
  void UnloadSector(int ix, int iy, int iz);
  uint32_t AllocSectorHandle();
  void FreeSectorHandle(uint32_t handle);
  void UploadIndirectionCell(int ix, int iy, int iz, uint32_t value);
  void UploadSectorInfo(uint32_t handle, const SectorInfo &info);
  void GenerateDetailBatch(const std::vector<SectorWorkItem> &sectors);
  void GenerateLODBatch(const std::vector<SectorWorkItem> &sectors);
  void RebuildSectorMasks();
};
