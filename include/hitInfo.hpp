#pragma once
#include "cumath.h"


struct hitInfo
{
    bool hit = false;
    float3 pos = make_float3(0, 0, 0);
    float3 normal = make_float3(0, 0, 0);
    float3 color = make_float3(0, 0, 0);
};