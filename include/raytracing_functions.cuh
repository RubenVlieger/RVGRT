#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "cuda_fp16.h"
#include "CoarseArray.cuh"


#define NUM_CONES 6
#define CONE_ANGLE 0.4f       // Radians, controls the spread of the cone
#define GI_MAX_DISTANCE 64.0f // How far a cone will trace
#define GI_STEP_SIZE 1.5f     // Initial step size for cone marching

struct hitInfo 
{
    float3 pos;
    float3 normal;
    half2 uv;
    bool hit;
    int its;
};

__device__ __forceinline__ bool IsSolid(int3 p, const uint32_t* __restrict__ bits) {
    uint64_t index = toIndex(p);
    return ((bits[index >> 5] >> (index & 31)) & 1);
}

__device__ __forceinline__ int toCoarseIndex(float3 pos) {
    int cx = floorf(pos.x) / COARSENESSSDF;
    int cy = floorf(pos.y) / COARSENESSSDF;
    int cz = floorf(pos.z) / COARSENESSSDF;
    return cz * SDF_SIZEX * SDF_SIZEY + cy * SDF_SIZEX + cx;
}

__device__ __forceinline__ float getDistance(float3 pos, const unsigned char* __restrict__ csdf)
{
    int cx = (int)(floorf(pos.x) * (1.0f / COARSENESSSDF));
    int cy = (int)(floorf(pos.y) * (1.0f / COARSENESSSDF));
    int cz = (int)(floorf(pos.z) * (1.0f / COARSENESSSDF));

    cx = min(cx, (int)SDF_SIZEX - 1);
    cy = min(cy, (int)SDF_SIZEY - 1);
    cz = min(cz, (int)SDF_SIZEZ - 1);
    cx = max(cx, 0);
    cy = max(cy, 0);
    cz = max(cz, 0);
    int cidx = cz * SDF_SIZEX * SDF_SIZEY + cy * SDF_SIZEX + cx;

    unsigned char dist = csdf[cidx];
    return (float)dist;
}
__device__ __forceinline__ unsigned char getDistance(int3 pos, const unsigned char* __restrict__ csdf)
{
    int cx = pos.x / COARSENESSSDF;
    int cy = pos.y / COARSENESSSDF;
    int cz = pos.z / COARSENESSSDF;

    cx = min(cx, (int)SDF_SIZEX - 1);
    cy = min(cy, (int)SDF_SIZEY - 1);
    cz = min(cz, (int)SDF_SIZEZ - 1);
    cx = max(cx, 0);
    cy = max(cy, 0);
    cz = max(cz, 0);
    
    int cidx = cz * SDF_SIZEX * SDF_SIZEY + cy * SDF_SIZEX + cx;
    return csdf[cidx];
}

__device__ float3 approximateCSDF(float3 pos, float3 dir, const unsigned char* __restrict__ csdf);

__device__ hitInfo trace(float3 camPos, float3 camDir, half distance,
                        const uint32_t* __restrict__ bits, const unsigned char* __restrict__ csdf);

__device__ float3 traceCone(float3 pos, float3 dir, const float4* __restrict__ radianceVoxels,
                           const unsigned char* __restrict__ csdf);

__device__ float3 traceCone(float3 pos,
                            float3 dir,
                            const uchar4* __restrict__ GIdata,
                            const unsigned char* __restrict__ csdf);


__device__ float3 sampleSky(float3 dir, float3 sunDir);
__device__ float3 sampleTexture(half2 uv, float3 pos, cudaTextureObject_t texObj);
