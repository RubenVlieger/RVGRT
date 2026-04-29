#include <simd/simd.h>
#include <cstdio>

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
  float deltaTime;

  simd_int3 worldOrigin;
};

struct ExposureData {
  float sceneLuminance;
  float padding[3];
};

struct SectorInfo {
  uint32_t baseBrickIndex;
  uint32_t flags;
  uint64_t brickMask;
};

int main() {
    printf("sizeof(CameraData) = %zu\n", sizeof(CameraData));
    printf("  offset of unjitteredViewProjection = %zu\n", offsetof(CameraData, unjitteredViewProjection));
    printf("  offset of jitter = %zu\n", offsetof(CameraData, jitter));
    printf("sizeof(FrameData) = %zu\n", sizeof(FrameData));
    printf("  offset of time = %zu\n", offsetof(FrameData, time));
    printf("  offset of deltaTime = %zu\n", offsetof(FrameData, deltaTime));
    printf("  offset of worldOrigin = %zu\n", offsetof(FrameData, worldOrigin));
    printf("sizeof(ExposureData) = %zu\n", sizeof(ExposureData));
    printf("sizeof(SectorInfo) = %zu\n", sizeof(SectorInfo));
    return 0;
}
