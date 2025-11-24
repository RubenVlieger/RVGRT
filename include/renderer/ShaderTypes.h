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
};

struct FrameData {
    simd_float3 sunDirection;
    float time;
    // We can add jitter values here later if needed
};

#endif