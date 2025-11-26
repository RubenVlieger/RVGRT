#ifndef ShaderTypes_h
#define ShaderTypes_h

// The simd library provides C++ types like `simd_float3` that are
// directly memory-compatible with Metal's `float3`.
#include <simd/simd.h>

// This struct is now defined in a plain C++ header.
// Both your .metal file and your .mm file will include this.
struct CameraData {
    simd_float3 position;
    simd_float3 forward;
    simd_float3 right;
    simd_float3 up;

    simd_float4x4 unjitteredViewProjection;
    simd_float4x4 prevUnjitteredViewProjection;
    
    simd_float2 jitter; 
    simd_float2 padding; // 16-byte aligned
};

struct FrameData {
    simd_float3 sunDirection;
    float time;
};

#endif