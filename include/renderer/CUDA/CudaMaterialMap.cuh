#pragma once

#include "renderer/CUDA/CudaBrickPool.cuh"
#include "renderer/ShaderTypes.h"
#include <cstdint>
#include <mutex>
#include <vector>

/**
 * CudaMaterialMap — Manages the voxel world with streaming and LOD.
 *
 * Mirrors the Metal MaterialMap implementation exactly.
 *
 * Owns:
 *   - Indirection 3D texture (toroidal, wraps around camera) as flat buffer
 *   - SectorInfo buffer (per-sector metadata)
 *   - BrickPool (occupancy + data buffers)
 *   - Super-sector mask buffer
 *
 * Call UpdateStreaming() each frame with the camera position.
 * Returns true if new sectors were loaded (caller should reset temporal
 * accumulation).
 */
class CudaMaterialMap {
public:
  CudaMaterialMap();
  ~CudaMaterialMap();

  /// Initial world generation around spawn point.
  void GenerateDynamic();

  /// Per-frame streaming update. Returns true if any sectors changed.
  bool UpdateStreaming(simd_float3 cameraPos);

  // Getters for rendering bindings (device pointers)
  uint32_t* GetIndirectionPtr() const { return d_indirection; }
  SectorInfo* GetSectorBufferPtr() const { return d_sectors; }
  uint64_t* GetOccupancyPtr() const { return _brickPool.GetOccupancyPtr(); }
  uint8_t* GetDataPtr() const { return _brickPool.GetDataPtr(); }
  uint64_t* GetSectorMaskPtr() const { return d_sectorMasks; }

  /// Get the world origin (world-space coordinate of indirection cell (0,0,0))
  int3 GetWorldOrigin() const { return _worldOrigin; }
  
  /// Get indirection dimensions
  int GetIndirectionWidth() const { return _indW; }
  int GetIndirectionHeight() const { return _indH; }
  int GetIndirectionDepth() const { return _indD; }

private:
  // --- GPU Resources ---
  // L1: 3D Texture as flat buffer (R32Uint). Value = sector handle (index into SectorBuffer)
  uint32_t* d_indirection;

  // L2: Buffer of SectorInfo structs
  SectorInfo* d_sectors;

  // L3+L4: Brick data (owned by BrickPool)
  CudaBrickPool _brickPool;

  // Super-sector masks: one uint64_t per 4x4x4 group of sectors
  uint64_t* d_sectorMasks;

  // Work buffers for GPU compute kernels
  SectorWorkItem* d_workItems;
  uint64_t* d_analysisResults;
  size_t _workItemCapacity;
  BrickWorkItem* d_brickWorkList;
  size_t _brickWorkCapacity;

  // --- Async Compute Results ---
  struct AsyncResult {
    SectorWorkItem item;
    uint64_t brickMask;
  };
  std::mutex _asyncResultsMutex;
  std::vector<AsyncResult> _asyncResults;

  // --- Streaming State ---
  int3 _worldOrigin; // World-space coordinate of indirection cell (0,0,0)
  int3 _lastCameraSector; // Camera's sector position at last update
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
  std::vector<SectorInfo> _sectorInfoCPU; // CPU mirror of sector buffer (for updates)
  std::vector<uint32_t> _indirectionCPU; // CPU mirror of indirection texture

  // CUDA stream for async compute
  void* _cudaStream;  // cudaStream_t

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

  /// Rebuild super-sector masks for affected regions
  void RebuildSectorMasks();

  /// Dispatch async analysis on GPU
  void DispatchAsyncAnalysis(const std::vector<SectorWorkItem> &items);

  /// Process completed async results
  void ProcessAsyncResults(std::vector<BrickWorkItem> &workList);

  /// Generate brick data on GPU
  void GenerateBrickData(const std::vector<BrickWorkItem> &workList);
};
