#pragma once
#include "cumath.h"


struct hitInfo
{
    bool hit = false;
    float3 pos = make_float3(0, 0, 0);
    half3 normal = make_half3(0.f);
    half3 color = make_half3(0.f);
};