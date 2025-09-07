#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "cuda_fp16.h"
#include "CoarseArray.cuh"
#include "raytracing_functions.cuh"


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
                    currentPos = posOnRay + camDir * ((float)dist * COARSENESSSDF);
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
/**
 * @brief Traces a single cone through the GI data and distance fields to gather indirect light.
 *
 * @param pos The starting position of the cone trace (i.e., the primary hit point).
 * @param dir The direction of the cone.
 * @param GIdata The pre-computed 3D grid of global illumination data.
 * @param csdf The coarse signed distance field for occlusion checks.
 * @return float3 The accumulated indirect light color.
 */
__device__ float3 traceCone(float3 pos,
                            float3 dir,
                            const uchar4* __restrict__ GIdata,
                            const unsigned char* __restrict__ csdf)
{
    float3 accumulatedColor = make_float3(0.0f, 0.0f, 0.0f);
    float accumulatedAlpha = 0.0f; // Represents how occluded the cone is.

    float currentDist = GI_STEP_SIZE * 2.0f; // Start slightly away from the surface

    for (int i = 0; i < 20; ++i) // Limit steps to avoid infinite loops
    {
        // Stop if the cone is fully occluded or has traveled too far
        if (accumulatedAlpha > 0.99f || currentDist > GI_MAX_DISTANCE) {
            break;
        }

        float3 currentPos = pos + dir * currentDist;

        // 1. Check for occlusion using your existing CSDF
        // We multiply by the SDF coarseness to get a world-space distance
        float sceneDist = getDistance(currentPos, csdf) * COARSENESSSDF;

        // 2. Calculate the cone's current width (radius)
        float coneWidth = currentDist * tanf(CONE_ANGLE);

        // 3. If the nearest surface is closer than the cone's width, it's an occlusion
        if (sceneDist < coneWidth)
        {
            // For simplicity, we stop the cone entirely.
            accumulatedAlpha = 1.0f;
            continue; // Go to the next iteration, which will break the loop
        }

        // 4. Sample the GI grid if we're not occluded
        int gx = (int)(floorf(currentPos.x) / COARSENESSGI);
        int gy = (int)(floorf(currentPos.y) / COARSENESSGI);
        int gz = (int)(floorf(currentPos.z) / COARSENESSGI);

        // Bounds check for the coarse GI grid
        if (gx >= 0 && gx < GI_SIZEX && gy >= 0 && gy < GI_SIZEY && gz >= 0 && gz < GI_SIZEZ)
        {
            uint64_t gidx = (uint64_t)gz * GI_SIZEX * GI_SIZEY + (uint64_t)gy * GI_SIZEX + gx;
            uchar4 giSample = GIdata[gidx];

            // Convert uchar4 to float3 color
            float3 voxelColor = make_float3(giSample.x / 255.0f, giSample.y / 255.0f, giSample.z / 255.0f);
            float voxelAlpha = giSample.w / 255.0f; // Assuming alpha is used for something like emission intensity

            // Blend the sampled color into our accumulated color
            float blendFactor = (1.0f - accumulatedAlpha) * voxelAlpha;
            accumulatedColor = accumulatedColor + voxelColor * blendFactor;
            accumulatedAlpha += blendFactor;
        }

        // 5. March the cone forward
        // The step size is proportional to the cone's current width to avoid missing details
        currentDist += max(GI_STEP_SIZE, coneWidth * 0.5f);
    }

    return accumulatedColor;
}


// /**
//  * This file contains the modified computeColor function and a new helper function 
//  * traceCone, which implements Voxel Cone Tracing for global illumination.
//  * * PREREQUISITE:
//  * This code assumes the existence of a 3D Radiance Voxel Grid named 'radianceVoxels'.
//  * This grid must be populated in a prior compute pass (the "Light Injection" pass)
//  * where direct lighting is calculated for each emissive voxel in the scene.
//  * It should have the same coarse dimensions as your `csdf`.
//  */

// // --- New VCT Constants (tweak these for performance vs. quality) ---
// #define NUM_CONES 6
// #define CONE_ANGLE 0.4f // Radians, controls the spread of the cone
// #define GI_MAX_DISTANCE 64.0f // How far a cone will trace
// #define GI_STEP_SIZE 1.5f     // Initial step size for cone marching

// // --- Forward declaration for the new radiance grid parameter ---
// // This would likely be a cudaTextureObject_t for a 3D texture in a real implementation
// // For simplicity, we'll pass it as a float4 pointer.
// // The float4 stores {R, G, B, Opacity}
// struct float4;


// // =================================================================================
// // NEW HELPER FUNCTION: traceCone
// // =================================================================================
// /**
//  * @brief Traces a single cone through the radiance and distance fields to gather indirect light.
//  * * @param pos The starting position of the cone trace (i.e., the primary hit point).
//  * @param dir The direction of the cone.
//  * @param radianceVoxels The pre-computed 3D grid of scene radiance.
//  * @param csdf The coarse signed distance field for occlusion checks.
//  * @return float3 The accumulated indirect light color.
//  */
// __device__ float3 traceCone(float3 pos,
//                             float3 dir,
//                             const uchar4* __restrict__ radianceVoxels,
//                             const unsigned char* __restrict__ csdf)
// {
//     float3 accumulatedColor = make_float3(0.0f, 0.0f, 0.0f);
//     float accumulatedAlpha = 0.0f; // Represents how occluded the cone is.
    
//     float currentDist = GI_STEP_SIZE * 2.0f; // Start slightly away from the surface to avoid self-occlusion
//     float3 currentPos = pos + dir * currentDist;

//     for (int i = 0; i < 20; ++i) // Limit steps to avoid infinite loops
//     {
//         // Stop if the cone is fully occluded or has traveled too far
//         if (accumulatedAlpha > 0.99f || currentDist > GI_MAX_DISTANCE) {
//             break;
//         }

//         // 1. Check for occlusion using your existing CSDF
//         float sceneDist = getDistance(currentPos, csdf) * COARSENESSGI;
        
//         // 2. Calculate the cone's current width (radius)
//         float coneWidth = currentDist * tanf(CONE_ANGLE);

//         // 3. If the nearest surface is closer than the cone's width, it's an occlusion
//         if (sceneDist < coneWidth)
//         {
//             // For simplicity, we stop the cone entirely.
//             // A more advanced technique would partially occlude based on how much `sceneDist` covers `coneWidth`.
//             accumulatedAlpha = 1.0f;
//             continue; // Go to the next iteration, which will break the loop
//         }

//         // 4. Sample the radiance grid if we're not occluded
//         int cx = (int)(floorf(currentPos.x) * (1.0f / COARSENESSGI));
//         int cy = (int)(floorf(currentPos.y) * (1.0f / COARSENESSGI));
//         int cz = (int)(floorf(currentPos.z) * (1.0f / COARSENESSGI));
        
//         // Bounds check for the coarse grid
//         if (cx >= 0 && cx < GI_SIZEX && cy >= 0 && cy < GI_SIZEY && cz >= 0 && cz < GI_SIZEZ)
//         {
//             int cidx = cz * GI_SIZEX * GI_SIZEY + cy * GI_SIZEX + cx;
//             uchar4 voxelSample =  radianceVoxels[cidx];
            
//             float3 voxelColor = make_float3(voxelSample.x, voxelSample.y, voxelSample.z);
//             float voxelAlpha = voxelSample.w;

//             // Blend the sampled color into our accumulated color
//             // The amount of light we add is determined by the voxel's alpha and how much "space" is left in our cone
//             float blendFactor = (1.0f - accumulatedAlpha) * voxelAlpha;
//             accumulatedColor = accumulatedColor + voxelColor * blendFactor;
//             accumulatedAlpha += blendFactor;
//         }

//         // 5. March the cone forward
//         // The step size should be proportional to the cone's current width to avoid missing details
//         currentDist += max(GI_STEP_SIZE, coneWidth * 0.5f);
//         currentPos = pos + dir * currentDist;
//     }

//     return accumulatedColor;
// }