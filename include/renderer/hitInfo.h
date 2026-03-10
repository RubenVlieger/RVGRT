#pragma once
#include "cumath.h"

struct hitInfo
{
    float3 pos;
    half3 normal;
    half2 uv; 
    bool hit;
    int its;
    uint8_t matID;
};
