#include <metal_stdlib>
#include "cumath.h"
#include "TerrainGeneration.h"

using namespace metal;

kernel void CArray_fill_kernel(device uint* world_data [[buffer(0)]], uint wordIdx [[thread_position_in_grid]])
{
    uint64_t baseBit = (uint64_t)wordIdx * 32ull;
    uint w = 0u;

    for (uint bit_offset = 0; bit_offset < 32; ++bit_offset) {

        uint64_t bitIndex = baseBit + bit_offset;
        uint64_t z = bitIndex >> (SHIX + SHIY);
        uint64_t y = (bitIndex >> SHIX) & (uint64_t)MODY;
        uint64_t x = bitIndex & (uint64_t)MODX;

        float density = Evaluate((float)x, (float)y, (float)z);

        if (density > 0.7f) {
            w |= (1u << bit_offset);
        }
    }
    world_data[wordIdx] = w;
}