#import "renderer/BrickPool.hpp"
#import "State.hpp"
#import "renderer/MetalDevice.hpp"
#include <algorithm>

// Under ARC, we store id objects as __bridge_retained void* in construction
// and release them in the destructor. The getters return the raw void* which
// callers (also in .mm) cast back to id<MTLBuffer> with (__bridge id<MTLBuffer>).

namespace {
id<MTLDevice> get_pool_device() {
  return static_cast<MetalDevice *>(State::state.graphicsDevice.get())
      ->GetMetalDevice();
}
} // namespace

BrickPool::BrickPool() : _usedCount(0), _searchStart(0) {
  _capacity = BRICK_POOL_CAPACITY;

  id<MTLDevice> dev = get_pool_device();
  _deviceHandle = (__bridge_retained void*)dev;

  // Allocate occupancy buffer: 8 uint64_t per brick
  NSUInteger occSize = (NSUInteger)_capacity * 8 * sizeof(uint64_t);
  id<MTLBuffer> occBuf = [dev newBufferWithLength:occSize
                                          options:MTLResourceStorageModePrivate];
  occBuf.label = @"BrickPool_Occupancy";
  _occupancyBuffer = (__bridge_retained void*)occBuf;

  // Allocate data buffer: 512 uint8_t per brick
  NSUInteger dataSize = (NSUInteger)_capacity * 512 * sizeof(uint8_t);
  id<MTLBuffer> dataBuf = [dev newBufferWithLength:dataSize
                                           options:MTLResourceStorageModePrivate];
  dataBuf.label = @"BrickPool_Data";
  _dataBuffer = (__bridge_retained void*)dataBuf;

  // Initialize all bricks as free
  _isFree.resize(_capacity, 1);

  NSLog(@"[BrickPool] Initialized: %u brick capacity (%.1f MB occupancy + %.1f "
        @"MB data = %.1f MB total)",
        _capacity, occSize / (1024.0 * 1024.0), dataSize / (1024.0 * 1024.0),
        (occSize + dataSize) / (1024.0 * 1024.0));
}

BrickPool::~BrickPool() {
  // Release the retained Objective-C objects
  if (_occupancyBuffer) { (void)(__bridge_transfer id)_occupancyBuffer; _occupancyBuffer = nullptr; }
  if (_dataBuffer) { (void)(__bridge_transfer id)_dataBuffer; _dataBuffer = nullptr; }
  if (_deviceHandle) { (void)(__bridge_transfer id)_deviceHandle; _deviceHandle = nullptr; }
}

uint32_t BrickPool::Allocate(uint32_t count) {
  if (count == 0)
    return UINT32_MAX;
  if (_capacity - _usedCount < count) {
    NSLog(@"[BrickPool] ALLOCATION FAILED: requested %u bricks, only %u free",
          count, _capacity - _usedCount);
    return UINT32_MAX;
  }

  uint32_t runLen = 0;
  uint32_t runStart = 0;

  // Search from _searchStart to end
  for (uint32_t i = _searchStart; i < _capacity; i++) {
    if (_isFree[i]) {
      if (runLen == 0)
        runStart = i;
      runLen++;
      if (runLen == count) {
        for (uint32_t j = runStart; j < runStart + count; j++)
          _isFree[j] = 0;
        _searchStart = (runStart + count) % _capacity;
        _usedCount += count;
        return runStart;
      }
    } else {
      runLen = 0;
    }
  }

  // Wrap around and search from 0 to _searchStart
  runLen = 0;
  for (uint32_t i = 0; i < _searchStart; i++) {
    if (_isFree[i]) {
      if (runLen == 0)
        runStart = i;
      runLen++;
      if (runLen == count) {
        for (uint32_t j = runStart; j < runStart + count; j++)
          _isFree[j] = 0;
        _searchStart = (runStart + count) % _capacity;
        _usedCount += count;
        return runStart;
      }
    } else {
      runLen = 0;
    }
  }

  NSLog(@"[BrickPool] ALLOCATION FAILED: no contiguous run of %u found", count);
  return UINT32_MAX;
}

void BrickPool::Free(uint32_t base, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    _isFree[base + i] = 1;
  }
  _usedCount -= count;
}

void* BrickPool::GetOccupancyBuffer() { return _occupancyBuffer; }
void* BrickPool::GetDataBuffer() { return _dataBuffer; }
