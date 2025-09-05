#pragma once
#include "StateRender.cuh"
#include "CArray.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "csdf.cuh"
#include "cuda_fp16.h"


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
    int cx = floorf(pos.x) / COARSENESS;
    int cy = floorf(pos.y) / COARSENESS;
    int cz = floorf(pos.z) / COARSENESS;
    return cz * C_SIZEX * C_SIZEY + cy * C_SIZEX + cx;
}

__device__ __forceinline__ float getDistance(float3 pos, const unsigned char* __restrict__ csdf)
{
    int cx = (int)(floorf(pos.x) * (1.0f / COARSENESS));
    int cy = (int)(floorf(pos.y) * (1.0f / COARSENESS));
    int cz = (int)(floorf(pos.z) * (1.0f / COARSENESS));

    cx = min(cx, (int)C_SIZEX - 1);
    cy = min(cy, (int)C_SIZEY - 1);
    cz = min(cz, (int)C_SIZEZ - 1);
    cx = max(cx, 0);
    cy = max(cy, 0);
    cz = max(cz, 0);
    int cidx = cz * C_SIZEX * C_SIZEY + cy * C_SIZEX + cx;

    unsigned char dist = csdf[cidx];
    return (float)dist;
}
__device__ __forceinline__ unsigned char getDistance(int3 pos, const unsigned char* __restrict__ csdf)
{
    int cx = pos.x / COARSENESS;
    int cy = pos.y / COARSENESS;
    int cz = pos.z / COARSENESS;

    cx = min(cx, (int)C_SIZEX - 1);
    cy = min(cy, (int)C_SIZEY - 1);
    cz = min(cz, (int)C_SIZEZ - 1);
    cx = max(cx, 0);
    cy = max(cy, 0);
    cz = max(cz, 0);
    
    int cidx = cz * C_SIZEX * C_SIZEY + cy * C_SIZEX + cx;
    return csdf[cidx];
}



__device__ float3 approximateCSDF(float3 pos, float3 dir, const unsigned char* __restrict__ csdf)
{
    int iterations = 0;
    while (iterations < 100) 
    {
        if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
            pos.x >= SIZEX || pos.y >= SIZEY || pos.z >= SIZEZ) return make_float3(-100.0f, -100.0f, -100.0f);

        //float dist = getDistance(pos, csdf);
        float dist = getDistance(pos, csdf);

        if (dist <= 1.0f) 
            return pos;
        
        pos = pos + dir * dist;
        iterations++;
    }
    return pos;
}

__device__ hitInfo trace(float3 camPos, float3 camDir,
                         half distance,
                         const uint32_t* __restrict__ bits,
                         const unsigned char* __restrict__ csdf) 
{
    float3 currentPos = camPos + distance * camDir; // unchanged march origin
    hitInfo HI; HI.hit = false;
    HI.its = 0.0f;

    float3 deltaDist = make_float3(
        (camDir.x != 0) ? fabsf(1.0f / camDir.x) : 1e10f,
        (camDir.y != 0) ? fabsf(1.0f / camDir.y) : 1e10f,
        (camDir.z != 0) ? fabsf(1.0f / camDir.z) : 1e10f);

    int3 step = make_int3((camDir.x > 0) - (camDir.x < 0),
                         (camDir.y > 0) - (camDir.y < 0),
                         (camDir.z > 0) - (camDir.z < 0));

    bool jumped = false;
    for (int majorIteration = 0; majorIteration < 5; majorIteration++) 
    {
        HI.its++;
        // Phase 1: CSDF approximation to get close to surfaces
        // pass camPos so approximateCSDF can map samples into CSDF space
        currentPos = approximateCSDF(currentPos, camDir, csdf);
        
        // Map currentPos into CSDF/voxel space for voxel indexing and distance queries
        // Phase 2: DDA for precise intersection (we compute ipos from mappedPos)
        int3 ipos = make_int3(floorf(currentPos.x), floorf(currentPos.y), floorf(currentPos.z));
        
        float3 tMax = make_float3(
            ((step.x > 0) ? (ipos.x + 1.0f - currentPos.x) : (currentPos.x - ipos.x)) * deltaDist.x,
            ((step.y > 0) ? (ipos.y + 1.0f - currentPos.y) : (currentPos.y - ipos.y)) * deltaDist.y,
            ((step.z > 0) ? (ipos.z + 1.0f - currentPos.z) : (currentPos.z - ipos.z)) * deltaDist.z
        );
        
        char mask = -128;
        for (int i = 0; i < 200; i++) {
            HI.its++;
            // Occasionally check for empty space and jump (but use mapped voxel distances)
            if ((i & 7) == 7) 
            {
                unsigned char dist = getDistance(ipos, csdf); // your existing getDistance overload works with ipos indexing
                if (dist > 2) {
                    // compute a point inside the voxel which lies along the path, using mapped space center
                    float3 voxelCenter = make_float3(ipos.x + 0.5f, ipos.y + 0.5f, ipos.z + 0.5f);
       
                    float t = dot(voxelCenter - currentPos, camDir);
                    float3 posOnRay = currentPos + t * camDir;

                    // Jump in rendered space along camDir
                    currentPos = posOnRay + camDir * ((float)dist * COARSENESS);
                    jumped = true;
                    break; // restart DDA from new advanced position
                }
            }
            
            // bounds check on mapped indices (voxel space)
            if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 ||
                ipos.x >= SIZEX || ipos.y >= SIZEY || ipos.z >= SIZEZ) {
                return HI; // Out of bounds
            }
            
            if (IsSolid(ipos, bits)) {
                HI.hit = true;
                // Determine intersection normal based on which axis we stepped through.
                // Compute intersection position using tMax values (we use values derived from mappedPos)
                if (mask == 0) {
                    HI.normal = make_float3(-step.x, 0, 0);
                    HI.pos = currentPos + (tMax.x - deltaDist.x) * camDir;
                    HI.uv = make_half2((HI.pos.y - (float)ipos.y), HI.pos.z - ipos.z);
                    HI.uv.y = step.x == -1 ? (half)1.0f - HI.uv.y : HI.uv.y;
                } else if (mask == 1) {
                    HI.normal = make_float3(0, -step.y, 0);
                    HI.pos = currentPos + (tMax.y - deltaDist.y) * camDir;
                    HI.uv = make_half2(HI.pos.x - ipos.x, HI.pos.z - ipos.z);
                } else if (mask == 2) {
                    HI.normal = make_float3(0, 0, -step.z);
                    HI.pos = currentPos + (tMax.z - deltaDist.z) * camDir;
                    HI.uv = make_half2(HI.pos.x - ipos.x, HI.pos.y - ipos.y);
                    HI.uv.x = step.z == 1 ? (half)1.0f - HI.uv.x : HI.uv.x;
                }
                return HI;
            }
            
            // DDA step (unchanged)
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) { 
                    tMax.x += deltaDist.x; 
                    ipos.x += step.x; 
                    mask = 0; 
                } else { 
                    tMax.z += deltaDist.z; 
                    ipos.z += step.z; 
                    mask = 2; 
                }
            } else {
                if (tMax.y < tMax.z) { 
                    tMax.y += deltaDist.y; 
                    ipos.y += step.y; 
                    mask = 1; 
                } else { 
                    tMax.z += deltaDist.z; 
                    ipos.z += step.z; 
                    mask = 2; 
                }
            }
        }
        if(jumped) {
            jumped = false;
            continue;
        }
        
        if (!HI.hit) break;
    }
    return HI;
}
