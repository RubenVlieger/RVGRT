#pragma once

#include "renderer/ShaderTypes.h"
#include <cstdint>
#include <vector>

/**
 * BrickPool — Fixed-capacity GPU allocator for 8x8x8 voxel bricks.
 *
 * Manages two GPU buffers:
 *   - Occupancy: 8 × uint64_t per brick (sub-brick masks)
 *   - Data: 512 × uint8_t per brick (material IDs)
 *
 * Allocation and deallocation are CPU-side (free-stack).
 * The GPU buffers are pre-allocated at max capacity.
 *
 * Platform-specific implementations live in BrickPool.mm (Metal)
 * and CudaBrickPool.cu (CUDA).
 */
class BrickPool {
public:
  BrickPool();
  ~BrickPool();

  /// Allocate `count` brick slots. Returns the base index, or UINT32_MAX on
  /// failure. Allocated slots are contiguous in the pool.
  uint32_t Allocate(uint32_t count);

  /// Free `count` brick slots starting at `base`.
  void Free(uint32_t base, uint32_t count);

  /// Get the occupancy buffer (capacity * 8 * sizeof(uint64_t))
  /// On Metal: returns id<MTLBuffer>, on CUDA: returns uint64_t* (device ptr)
  void* GetOccupancyBuffer();

  /// Get the data buffer (capacity * 512 * sizeof(uint8_t))
  /// On Metal: returns id<MTLBuffer>, on CUDA: returns uint8_t* (device ptr)
  void* GetDataBuffer();

  uint32_t GetCapacity() const { return _capacity; }
  uint32_t GetUsedCount() const { return _usedCount; }
  uint32_t GetFreeCount() const { return _capacity - _usedCount; }

private:
  void* _deviceHandle; // Platform-specific device (id<MTLDevice> or opaque)
  void* _occupancyBuffer;
  void* _dataBuffer;

  uint32_t _capacity;
  uint32_t _usedCount;

  // Next-Fit bitset style array for O(1) amortized allocation
  std::vector<uint8_t> _isFree;
  uint32_t _searchStart;
};
