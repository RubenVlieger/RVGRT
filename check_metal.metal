#include <metal_stdlib>
using namespace metal;

struct CameraData {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float4x4 unjitteredViewProjection;
    float4x4 prevUnjitteredViewProjection;
    float2 jitter;
    float2 padding;
};

struct FrameData {
    float3 sunDirection;
    float time;
    float deltaTime;
    int3 worldOrigin;
};

kernel void testKernel(uint3 gid [[thread_position_in_grid]],
                       constant CameraData& cam [[buffer(0)]],
                       constant FrameData& frame [[buffer(1)]]) {
    float3 p = cam.position + cam.forward + cam.right + cam.up;
    float3 s = frame.sunDirection;
    float t = frame.time + frame.deltaTime;
    int3 w = frame.worldOrigin;
}
