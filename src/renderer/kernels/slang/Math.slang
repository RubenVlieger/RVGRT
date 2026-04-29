#pragma once

// Math helpers, constants, and texture atlas indices for Slang shaders

#include "Settings.slang"

static const float PI = 3.1415926535f;

static const half3 c_sunColor = half3(4.5h, 3.6h, 3.0h);

float3 SampleSky(float3 dir, float3 sunDir);

uint64_t toIndex(int3 p)
{
    return (((uint64_t)p.x) & MODX) | ((((uint64_t)p.y) & MODY) << SHIX) |
           ((((uint64_t)p.z) & MODZ) << (SHIX + SHIY));
}

uint64_t toIndex(uint64_t x, uint64_t y, uint64_t z)
{
    return (x & MODX) | ((y & MODY) << SHIX) | ((z & MODZ) << (SHIX + SHIY));
}