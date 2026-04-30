#pragma once
#ifndef __SLANG__

#if defined(__APPLE__)
#include <simd/simd.h>
#elif defined(__EMSCRIPTEN__)
#include <cstdint>
#include <glm/glm.hpp>
using simd_float2 = glm::vec2;
using simd_float3 = glm::vec3;
using simd_float4 = glm::vec4;
using simd_int3 = glm::ivec3;
using simd_float4x4 = glm::mat4;
inline simd_float3 simd_normalize(simd_float3 v) { return glm::normalize(v); }
inline simd_float3 simd_make_float3(float x, float y, float z) { return glm::vec3(x, y, z); }
inline simd_float2 simd_make_float2(float x, float y) { return glm::vec2(x, y); }
#else
#include <cstdint>
#include "../cumath.h"
using simd_float2 = float2;
using simd_float3 = float3;
using simd_float4 = float4;
using simd_int3 = int3;
using simd_float4x4 = mat4;
inline float3 simd_normalize(float3 v) { return normalize(v); }
inline float3 simd_make_float3(float x, float y, float z) { return make_float3(x, y, z); }
inline float2 simd_make_float2(float x, float y) { return make_float2(x, y); }
#endif

#define SHADER_TYPES_MATH_INCLUDED

#endif
