#pragma once
#include "cumath.h"

struct hitInfo
{
    float3 pos;
#if defined(PLATFORM_CUDA)
    float3 normal;
    float2 uv;
#else
    half3 normal;
    half2 uv;
#endif
    bool hit;
    int its;
    uint8_t matID;
};
