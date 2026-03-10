#pragma once

#include "renderer/BrickPoolTraits.hpp"
#include "renderer/ShaderTypes.h"
#include <cstdint>
#include <vector>
#include <algorithm>

template<typename Traits>
class BrickPool {
public:
    BrickPool() : _usedCount(0), _searchStart(0) {
        _device = Traits::GetDevice();
        _capacity = BRICK_POOL_CAPACITY;

        size_t occSize = static_cast<size_t>(_capacity) * 8 * sizeof(uint64_t);
        size_t dataSize = static_cast<size_t>(_capacity) * 512 * sizeof(uint8_t);

        _occupancyBuffer = Traits::AllocateOccupancy(_device, occSize);
        _dataBuffer = Traits::AllocateData(_device, dataSize);

        Traits::ZeroOccupancy(_device, _occupancyBuffer, occSize);
        Traits::ZeroData(_device, _dataBuffer, dataSize);

        _isFree.resize(_capacity, 1);

        Traits::Log("Initialized: %u brick capacity (%.1f MB occupancy + %.1f MB data = %.1f MB total)",
                    _capacity, occSize / (1024.0 * 1024.0), dataSize / (1024.0 * 1024.0),
                    (occSize + dataSize) / (1024.0 * 1024.0));
    }

    ~BrickPool() {
#if defined(_WIN32)
        Traits::FreeBuffer(_occupancyBuffer);
        Traits::FreeBuffer(_dataBuffer);
#else
        _occupancyBuffer = nullptr;
        _dataBuffer = nullptr;
#endif
    }

    uint32_t Allocate(uint32_t count) {
        if (count == 0)
            return UINT32_MAX;
        if (_capacity - _usedCount < count) {
            Traits::LogError("ALLOCATION FAILED: requested %u bricks, only %u free",
                             count, _capacity - _usedCount);
            return UINT32_MAX;
        }

        uint32_t runLen = 0;
        uint32_t runStart = 0;

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

        Traits::LogError("ALLOCATION FAILED: no contiguous run of %u found", count);
        return UINT32_MAX;
    }

    void Free(uint32_t base, uint32_t count) {
        if (base >= _capacity || base + count > _capacity) {
            Traits::LogError("WARNING: Free out of bounds (base=%u, count=%u, capacity=%u)",
                             base, count, _capacity);
            return;
        }
        
        for (uint32_t i = 0; i < count; i++) {
            _isFree[base + i] = 1;
        }
        _usedCount -= count;
    }

    auto GetOccupancyBuffer() { return _occupancyBuffer; }
    auto GetDataBuffer() { return _dataBuffer; }
    
    auto GetOccupancyPtr() { return _occupancyBuffer; }
    auto GetDataPtr() { return _dataBuffer; }

    auto GetOccupancyPtr() const { return _occupancyBuffer; }
    auto GetDataPtr() const { return _dataBuffer; }

    uint32_t GetCapacity() const { return _capacity; }
    uint32_t GetUsedCount() const { return _usedCount; }
    uint32_t GetFreeCount() const { return _capacity - _usedCount; }

private:
    typename Traits::DeviceHandle _device;
    typename Traits::OccupancyBuffer _occupancyBuffer;
    typename Traits::DataBuffer _dataBuffer;

    uint32_t _capacity;
    uint32_t _usedCount;

    std::vector<uint8_t> _isFree;
    uint32_t _searchStart;
};

#if defined(__APPLE__)
using MetalBrickPool = BrickPool<BrickPoolImpl::MetalBrickPoolTraits>;
using PlatformBrickPool = MetalBrickPool;
#elif defined(_WIN32)
using CudaBrickPool = BrickPool<BrickPoolImpl::CudaBrickPoolTraits>;
using PlatformBrickPool = CudaBrickPool;
#endif
