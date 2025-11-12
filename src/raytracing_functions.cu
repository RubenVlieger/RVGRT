#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "cuda_fp16.h"
#include "CoarseArray.cuh"
#include "raytracing_functions.cuh"
#include "TerrainGeneration.cuh"

__device__ __forceinline__ bool IsSolid(const glm::ivec3& p, const uint32_t* __restrict__ bits) {
    // Note: The toIndex function still uses the shift constants from cumath.hpp
    uint64_t index = toIndex(p.x, p.y, p.z);
    return ((bits[index >> 5] >> (index & 31)) & 1);
}

__device__ glm::vec3 sampleSky(const glm::vec3& dir, const glm::vec3& sunDir)
{
    float sunDot = glm::dot(dir, sunDir);
    if (sunDot > 0.999f) {
        // Convert the constant float3 from cumath.hpp to a glm::vec3 for the return value
        return glm::vec3(c_sunColor.x, c_sunColor.y, c_sunColor.z);
    } else {
        float t = glm::clamp(0.5f * (dir.y + 1.0f), 0.0f, 1.0f);
        return glm::mix(glm::vec3(0.2f, 0.4f, 0.8f),   // horizon blue
                        glm::vec3(0.6f, 0.8f, 1.0f),   // zenith blue
                        t);
    }
}

__device__ glm::vec3 sampleTexture(half2 uv, const glm::vec3& pos, cudaTextureObject_t texObj)
{
    // Texture selection constants
    const half2 texStoneID = make_half2(0.0f / 16.0f, 1.0f / 16.0f);
    const half2 texDirtID = make_half2(0.0f / 16.0f, 2.0f / 16.0f);
    const half2 texCobbleID = make_half2(1.0f / 16.0f, 0.0f / 16.0f);
    const half2 texIronID = make_half2(2.0f / 16.0f, 1.0f / 16.0f);
    const half2 texDiamondID = make_half2(3.0f / 16.0f, 2.0f / 16.0f);
    const half2 texStone2ID = make_half2(0.0f / 16.0f, 0.0f / 16.0f);
    const half2 texSandStoneID = make_half2(11.0f / 16.0f, 0.0f / 16.0f);
    const half2 texCoalID = make_half2(2.0f / 16.0f, 2.0f / 16.0f);
    half2 whichBlock = make_half2(0.0f, 8.0f / 16.0f);

    // Voxel material selection based on 3D noise
    const float freq = 0.05f;
    glm::ivec3 ipos = glm::floor(pos);
    float eval = simplex3D(ipos.x * freq, ipos.y * freq, ipos.z * freq);
    float eval2 = simplex3D((ipos.x + 121.3f) * freq * 0.3f, (ipos.y + 1321.3f) * freq * 0.3f, (ipos.z + 721.5f) * freq * 0.3f);
    eval = eval * 0.4f + eval2 * 0.6f;

    if(eval < -1.3f) whichBlock = texStoneID;
    else if(eval < -1.2f) whichBlock = texDiamondID;
    else if(eval < -0.7f) whichBlock = texIronID;
    else if(eval < 0.0f) whichBlock = texStoneID;
    else if(eval < 0.1f) whichBlock = texCoalID;
    else if(eval < 0.4f) whichBlock = texCobbleID;
    else if(eval < 0.8f) whichBlock = texDirtID;
    else if(eval < 1.2f) whichBlock = texStone2ID;
    else whichBlock = texStoneID;

    // Calculate final UV in the atlas
    uv.x = ((uv.x * hrcp(16.0))) + whichBlock.x;
    uv.y = ((uv.y * hrcp(16.0))) + whichBlock.y;

    // Sample the texture and return as glm::vec3
    float4 t = tex2D<float4>(texObj, __half2float(uv.y), __half2float(uv.x));
    return glm::vec3(t.x, t.y, t.z);
}

__device__ float getDistance(const glm::vec3& pos, const unsigned char* __restrict__ csdf)
{
    glm::ivec3 c = glm::floor(pos / (float)COARSENESSSDF);
    c = glm::clamp(c, glm::ivec3(0), glm::ivec3(SDF_SIZEX - 1, SDF_SIZEY - 1, SDF_SIZEZ - 1));
    int cidx = c.z * SDF_SIZEX * SDF_SIZEY + c.y * SDF_SIZEX + c.x;
    return (float)csdf[cidx];
}

__device__ unsigned char getDistance(const glm::ivec3& pos, const unsigned char* __restrict__ csdf)
{
    glm::ivec3 c = pos / (int)COARSENESSSDF;
    c = glm::clamp(c, glm::ivec3(0), glm::ivec3(SDF_SIZEX - 1, SDF_SIZEY - 1, SDF_SIZEZ - 1));
    int cidx = c.z * SDF_SIZEX * SDF_SIZEY + c.y * SDF_SIZEX + c.x;
    return csdf[cidx];
}

__device__ glm::vec3 approximateCSDF(glm::vec3 pos, const glm::vec3& dir, const unsigned char* __restrict__ csdf)
{
    for(int i = 0; i < 100; ++i) {
        if (glm::any(glm::lessThan(pos, glm::vec3(0))) || glm::any(glm::greaterThanEqual(pos, glm::vec3(SIZEX, SIZEY, SIZEZ)))) {
            return glm::vec3(-100.0f);
        }
        float dist = getDistance(pos, csdf);
        if (dist <= 1.0f) return pos;
        pos += dir * dist;
    }
    return pos;
}

__device__ hitInfo trace(glm::vec3 camPos, const glm::vec3& camDir, half distance,
                         const uint32_t* __restrict__ bits, const unsigned char* __restrict__ csdf)
{
    hitInfo HI;
    HI.hit = false;
    HI.its = 0;
    HI.pos = glm::vec3(-500.0f);

    glm::vec3 currentPos = camPos + (float)distance * camDir;

    glm::vec3 deltaDist = glm::abs(glm::vec3(1.0f) / camDir);
    glm::ivec3 step = glm::sign(camDir);

    bool jumped = false;
    for (int majorIteration = 0; majorIteration < 5; majorIteration++)
    {
        HI.its++;
        currentPos = approximateCSDF(currentPos, camDir, csdf);
        
        glm::ivec3 ipos = glm::floor(currentPos);
        glm::vec3 tMax = (glm::vec3(ipos) - currentPos + (glm::vec3(step) * 0.5f) + 0.5f) * deltaDist;
        
        char mask = -128;
        for (int i = 0; i < 200; i++) {
            HI.its++;
            if ((i & 7) == 7) {
                unsigned char dist = getDistance(ipos, csdf);
                if (dist > 2) {
                    currentPos += camDir * ((float)dist * COARSENESSSDF);
                    jumped = true;
                    break;
                }
            }
            
            if (glm::any(glm::lessThan(ipos, glm::ivec3(0))) || glm::any(glm::greaterThanEqual(ipos, glm::ivec3(SIZEX, SIZEY, SIZEZ)))) {
                 return HI; // Out of bounds
            }
            
            if (IsSolid(ipos, bits)) {
                HI.hit = true;
                if (mask == 0) {
                    HI.normal = glm::vec3(-step.x, 0, 0);
                    HI.pos = currentPos + (tMax.x - deltaDist.x) * camDir;
                    HI.uv = make_half2(HI.pos.y - ipos.y, HI.pos.z - ipos.z);
                    if(step.x == -1) HI.uv.y = (half)1.0f - HI.uv.y;
                } else if (mask == 1) {
                    HI.normal = glm::vec3(0, -step.y, 0);
                    HI.pos = currentPos + (tMax.y - deltaDist.y) * camDir;
                    HI.uv = make_half2(HI.pos.x - ipos.x, HI.pos.z - ipos.z);
                } else { // mask == 2
                    HI.normal = glm::vec3(0, 0, -step.z);
                    HI.pos = currentPos + (tMax.z - deltaDist.z) * camDir;
                    HI.uv = make_half2(HI.pos.x - ipos.x, HI.pos.y - ipos.y);
                    if(step.z == 1) HI.uv.x = (half)1.0f - HI.uv.x;
                }
                return HI;
            }
            
            // DDA step
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) { tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; }
                else                 { tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; }
            } else {
                if (tMax.y < tMax.z) { tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; }
                else                 { tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; }
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

__device__ glm::vec3 traceCone(glm::vec3 pos, const glm::vec3& dir,
                               const uchar4* __restrict__ GIdata,
                               const unsigned char* __restrict__ csdf)
{
    glm::vec3 accumulatedColor(0.0f);
    float accumulatedAlpha = 0.0f;
    float currentDist = GI_STEP_SIZE * 2.0f;

    for (int i = 0; i < 20; ++i) {
        if (accumulatedAlpha > 0.99f || currentDist > GI_MAX_DISTANCE) break;

        glm::vec3 currentPos = pos + dir * currentDist;
        float sceneDist = getDistance(currentPos, csdf) * COARSENESSSDF;
        float coneWidth = currentDist * tanf(CONE_ANGLE);

        if (sceneDist < coneWidth) {
            accumulatedAlpha = 1.0f;
            continue;
        }

        glm::ivec3 g = glm::floor(currentPos / (float)COARSENESSGI);
        
        if (glm::all(glm::greaterThanEqual(g, glm::ivec3(0))) && glm::all(glm::lessThan(g, glm::ivec3(GI_SIZEX, GI_SIZEY, GI_SIZEZ))))
        {
            uint64_t gidx = (uint64_t)g.z * GI_SIZEX * GI_SIZEY + (uint64_t)g.y * GI_SIZEX + g.x;
            uchar4 giSample = GIdata[gidx];
            
            glm::vec3 voxelColor(giSample.x / 255.0f, giSample.y / 255.0f, giSample.z / 255.0f);
            float voxelAlpha = giSample.w / 255.0f;

            float blendFactor = (1.0f - accumulatedAlpha) * voxelAlpha;
            accumulatedColor += voxelColor * blendFactor;
            accumulatedAlpha += blendFactor;
        }
        currentDist += glm::max(GI_STEP_SIZE, coneWidth * 0.5f);
    }
    return accumulatedColor;
}