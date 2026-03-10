#pragma once
#include "cumath.h"


struct hitInfo
{
    bool hit = false;
    float3 pos = make_float3(0, 0, 0);
#if defined(PLATFORM_CUDA)
    float3 normal = make_float3(0.f);
    float3 color = make_float3(0.f);
#else
    half3 normal = make_half3(0.f);
    half3 color = make_half3(0.f);
#endif
};
