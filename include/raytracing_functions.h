#include "cumath.h" 
#include "CoarseArray.h"
class CoarseArray;

struct hitInfo
{
    float3 pos;
    half3 normal;
    half2 uv; 
    bool hit;
    int its;
};

/**
 * @brief Checks if a voxel at a given integer coordinate is solid.
 * @param p The integer coordinates (x, y, z) of the voxel.
 * @param bits Pointer to the packed voxel data on the GPU.
 * @return True if the voxel is solid, false otherwise.
 */




GPU_FUNC GPU_INLINE bool is_voxel_solid(int3 p, TEX3D_U32_R voxels)
{
    if (p.x < 0 || p.y < 0 || p.z < 0 || 
        p.x >= SIZEX || p.y >= SIZEY || p.z >= SIZEZ) return false;

#if defined(PLATFORM_METAL)
    uint packedByte = voxels.read(uint3(p.x >> 1, p.y >> 1, p.z >> 1)).r;
    uint bitIndex = (p.x & 1) | ((p.y & 1) << 1) | ((p.z & 1) << 2);
    return (packedByte >> bitIndex) & 1;
#else
    uint64_t index = toIndex(p.x, p.y, p.z);
    return ((voxels[index >> 5] >> (index & 31)) & 1);
#endif
}

GPU_FUNC GPU_INLINE float get_csdf_val(int3 c, TEX3D_U8_R csdf)
{
    c.x = max(0, min(c.x, (int)SDF_SIZEX - 1));
    c.y = max(0, min(c.y, (int)SDF_SIZEY - 1));
    c.z = max(0, min(c.z, (int)SDF_SIZEZ - 1));

#if defined(PLATFORM_METAL)
    return (float)csdf.read(uint3(c.x, c.y, c.z)).r;
#else
    int cidx = c.z * SDF_SIZEX * SDF_SIZEY + c.y * SDF_SIZEX + c.x;
    return (float)csdf[cidx];
#endif
}

GPU_FUNC GPU_INLINE uint32_t get_gi_val(int3 g, TEX3D_U32_R giData)
{
#if defined(PLATFORM_METAL)
    return giData.read(uint3(g.x, g.y, g.z)).r;
#else
    uint64_t gidx = (uint64_t)g.z * GI_SIZEX * GI_SIZEY + (uint64_t)g.y * GI_SIZEX + g.x;
    return giData[gidx];
#endif
}


GPU_FUNC GPU_INLINE bool IsSolid(const int3 p, TEX3D_U32_R bits)
{
    return is_voxel_solid(p, bits);
}


GPU_FUNC GPU_INLINE bool isCoarseBlockSolid(uint64_t cx, uint64_t cy, uint64_t cz, TEX3D_U32_R bits)
{
#if defined(PLATFORM_METAL)
    if (cx >= bits.get_width() || cy >= bits.get_height() || cz >= bits.get_depth()) return false;

    uint packedByte = bits.read(uint3(cx, cy, cz)).r;
    return packedByte > 0;
#else
    int3 start = make_int3(cx * COARSENESSSDF, cy * COARSENESSSDF, cz * COARSENESSSDF);
    int3 end = make_int3(start.x + COARSENESSSDF, start.y + COARSENESSSDF, start.z + COARSENESSSDF);

    for (int z = start.z; z < end.z; ++z) 
        for (int y = start.y; y < end.y; ++y) 
            for (int x = start.x; x < end.x; ++x) 
                if (IsSolid(make_int3(x, y, z), bits)) 
                    return true;       
    return false;
#endif
}


GPU_FUNC GPU_INLINE float getDistance(const float3 pos, TEX3D_U8_R csdf)
{
    float3 gridPos = pos * (1.0f / (float)COARSENESSSDF);
    
    float3 uvw = (gridPos + 0.5f) / float3(SDF_SIZEX, SDF_SIZEY, SDF_SIZEZ);

#if defined(PLATFORM_METAL)
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    
    float val = csdf.sample(s, uvw).r;
    
    return val;
#else
    return get_csdf_val(gridPos, csdf);
#endif
}



GPU_FUNC float traceDistanceOnly(float3 camPos, float3 camDir, float maxDist, TEX3D_U8_R csdf)
{
    float t = 0.0f;
    float3 pos = camPos;

    for(int i = 0; i < 24; ++i) 
    {
        if (t >= maxDist) return maxDist;

        // Check bounds
        if (pos.x < 0.0f || pos.y < 0.0f || pos.z < 0.0f || 
            pos.x >= (float)SIZEX || pos.y >= (float)SIZEY || pos.z >= (float)SIZEZ) {
            return maxDist;
        }
        float coarseDist = getDistance(pos, csdf);
        
        // Convert to World Units
        float distWorld = (coarseDist - 0.5f) * (float)COARSENESSSDF;

        if (distWorld < 1.0f) {
            return t; // Treat as hit
        }
        t += distWorld;
        pos = camPos + camDir * t;
    }
    
    return t;
}


GPU_FUNC float traceShadowCSDF(float3 pos, float3 lightDir, float maxDist, TEX3D_U8_R csdf)
{
    float t = 1.f * (float)COARSENESSSDF; 
    float res = 1.0f;
    const float k = 4.0f; // Softness factor (lower = softer)

    for(int i = 0; i < 24; ++i)
    {
        if (t >= maxDist) return res;

        float3 p = pos + lightDir * t;
        
        if (p.x < 0 || p.y < 0 || p.z < 0 || 
            p.x >= SIZEX || p.y >= SIZEY || p.z >= SIZEZ) {
            return res; 
        }

        float d = getDistance(p, csdf);

        float dWorld = (d - 0.5f) * (float)COARSENESSSDF;
        if (dWorld < 0.1f) {
            return 0.1f; // Hit something -> Occluded
        }

        // Penumbra calculation (Soft Shadow)
        // As we get closer to objects, the shadow gets darker
        res = fmin(res, k * dWorld / t);

        t += dWorld;
    }
    return res;
}

GPU_FUNC float3 approximateCSDF(float3 pos, const float3 dir, TEX3D_U8_R csdf)
{
    for(int i = 0; i < 48; ++i) {
        if (pos.x < 0.0f || pos.y < 0.0f || pos.z < 0.0f || 
            pos.x >= (float)SIZEX || pos.y >= (float)SIZEY || pos.z >= (float)SIZEZ) {
            return make_float3(-1000000.0f);
        }

        float dist = getDistance(pos, csdf);
                
        float fineDist = (dist - 0.5f) * (float)COARSENESSSDF;        
        
        if (fineDist <= 1.25f) return pos; 
        
        pos = pos + dir * fineDist;

    }
    return pos;
}


GPU_FUNC GPU_INLINE int3 clamp3(const int3 a, const int3 b, const int3 c)
{
    int3 d = make_int3(a.x < b.x ? b.x : a.x,
                       a.y < b.y ? b.y : a.y,
                       a.z < b.z ? b.z : a.z);
    return make_int3(d.x > c.x ? c.x : d.x,
                     d.y > c.y ? c.y : d.y,
                     d.z > c.z ? c.z : d.z);
}

GPU_FUNC GPU_INLINE half3 clamp3(const half3 a, const half3 b, const half3 c)
{
    half3 d = make_half3(a.x < b.x ? b.x : a.x,
                       a.y < b.y ? b.y : a.y,
                       a.z < b.z ? b.z : a.z);
    return make_half3(d.x > c.x ? c.x : d.x,
                     d.y > c.y ? c.y : d.y,
                     d.z > c.z ? c.z : d.z);
}


/**
 * @brief The main hybrid ray tracing function combining CSDF marching and DDA.
 * @param camPos The starting position of the ray.
 * @param camDir The normalized direction of the ray.
 * @param distance An initial distance to advance the ray before tracing begins.
 * @param bits Pointer to the packed high-resolution voxel data.
 * @param csdf Pointer to the CSDF data.
 * @return A hitInfo struct containing the results of the trace.
 */
GPU_FUNC hitInfo trace(float3 camPos, 
                       const float3 camDir, 
                       float distance,
                       TEX3D_U32_R bits, 
                       TEX3D_U8_R csdf)
{
    hitInfo HI;
    HI.hit = false;
    HI.its = 0;
    HI.pos = make_float3(-5000.f); 

    // Move to start distance
    float3 currentPos = camPos + camDir * distance;

    // Precompute DDA constants (invariant for the ray)
    const float3 deltaDist = make_float3(
        abs(camDir.x) > 1e-5f ? abs(1.0f / camDir.x) : 1.0e30f,
        abs(camDir.y) > 1e-5f ? abs(1.0f / camDir.y) : 1.0e30f,
        abs(camDir.z) > 1e-5f ? abs(1.0f / camDir.z) : 1.0e30f
    );

    const int3 step = make_int3(
        camDir.x > 0.0f ? 1 : -1,
        camDir.y > 0.0f ? 1 : -1,
        camDir.z > 0.0f ? 1 : -1
    );

    for (int majorIteration = 0; majorIteration < 10; majorIteration++)
    {
        currentPos = approximateCSDF(currentPos, camDir, csdf);

        int3 ipos = to_int3(floor3(currentPos));
        float3 fpos = make_float3(ipos);
        
        float3 tMax;
        tMax.x = ((step.x > 0) ? (fpos.x + 1.0f - currentPos.x) : (currentPos.x - fpos.x)) * deltaDist.x;
        tMax.y = ((step.y > 0) ? (fpos.y + 1.0f - currentPos.y) : (currentPos.y - fpos.y)) * deltaDist.y;
        tMax.z = ((step.z > 0) ? (fpos.z + 1.0f - currentPos.z) : (currentPos.z - fpos.z)) * deltaDist.z;

        int mask = -1;
        float distTraveledInDDA = 0.0f;
        bool hitFound = false;

        for (int i = 0; i < 8; i++) 
        {
            HI.its++;
            
            if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 || 
                ipos.x >= (int)SIZEX || ipos.y >= (int)SIZEY || ipos.z >= (int)SIZEZ) {
                return HI;
            }

            if (IsSolid(ipos, bits)) {
                hitFound = true;
                break;
            }
            
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) { 
                    distTraveledInDDA = tMax.x;
                    tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; 
                } else { 
                    distTraveledInDDA = tMax.z;
                    tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
                }
            } else {
                if (tMax.y < tMax.z) { 
                    distTraveledInDDA = tMax.y;
                    tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; 
                } else { 
                    distTraveledInDDA = tMax.z;
                    tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
                }
            }
        }
        if (hitFound) {
            HI.hit = true;
            float tVal = 0.0f;

            if (mask == -1) //if started inside a solid block
            {
                float tBackX = deltaDist.x - tMax.x;
                float tBackY = deltaDist.y - tMax.y;
                float tBackZ = deltaDist.z - tMax.z;

                // Find the smallest distance backwards (closest face behind us)
                if (tBackX < tBackY && tBackX < tBackZ) {
                    mask = 0;
                    tVal = -tBackX;
                } else if (tBackY < tBackZ) {
                    mask = 1;
                    tVal = -tBackY;
                } else {
                    mask = 2;
                    tVal = -tBackZ;
                }
            }

            if (mask == 0) {
                tVal = tMax.x - deltaDist.x;
                HI.normal = make_half3(-(half)step.x, 0.h, 0.h);
            } else if (mask == 1) {
                tVal = tMax.y - deltaDist.y;
                HI.normal = make_half3(0.h, -(half)step.y, 0.h);
            } else {
                tVal = tMax.z - deltaDist.z;
                HI.normal = make_half3(0.h, 0.h, -(half)step.z);
            }
            
            HI.pos = currentPos + camDir * tVal;

            float3 fpos_hit = floor3(HI.pos);
            if (mask == 0) {
                HI.uv = make_half2(HI.pos.y - fpos_hit.y, HI.pos.z - fpos_hit.z);
                if(step.x == -1) HI.uv.y = 1.0h - HI.uv.y;
            } else if (mask == 1) {
                HI.uv = make_half2(HI.pos.x - fpos_hit.x, HI.pos.z - fpos_hit.z);
            } else {
                HI.uv = make_half2(HI.pos.x - fpos_hit.x, HI.pos.y - fpos_hit.y);
                if(step.z == 1) HI.uv.x = 1.0h - HI.uv.x;
            }
            return HI;
        }
        currentPos += camDir * (distTraveledInDDA + 0.0001f);        
    }
    return HI;
}




/**
 * @brief Traces a cone through the GI data grid to gather indirect illumination.
 * @param pos The starting point of the cone (a surface point).
 * @param dir The central direction of the cone.
 * @param GIdata Pointer to the 3D grid of GI voxel data.
 * @param csdf Pointer to the CSDF for occlusion checks.
 * @return A float3 color representing the accumulated indirect light.
 */
GPU_FUNC float3 traceCone(float3 pos, const float3 dir, 
                          TEX3D_U32_R GIdata,
                          TEX3D_U8_R csdf)
{
    float3 accumulatedColor = make_float3(0.0f);
    float accumulatedAlpha = 0.0f;
    float currentDist = COARSENESSGI * 2.0f;

    for (int i = 0; i < 20; ++i) {
        if (accumulatedAlpha > 0.99f || currentDist > GI_SIZEX) break;

        float3 currentPos = pos + dir * currentDist;
        float sceneDist = getDistance(currentPos, csdf) * COARSENESSSDF;
        float coneWidth = currentDist;

        if (sceneDist < coneWidth) {
            accumulatedAlpha = 1.0f;
            continue;
        }
        int3 g = to_int3(floor3(currentPos / (float)COARSENESSGI));
        
        if (!((g.x < 0)              | (g.y < 0)                   | (g.z < 0) ||
         ((uint64_t)g.x >= GI_SIZEX) | ((uint64_t)g.y >= GI_SIZEY) | ((uint64_t)g.z >= GI_SIZEZ)))
        {       
            uint32_t giSample = get_gi_val(g, GIdata);
            
            float3 voxelColor = make_float3((giSample&255) / 255.0f, ((giSample>>8)&255) / 255.0f, ((giSample>>16)&255) / 255.0f);
            float voxelAlpha = ((giSample>>24)&255) / 255.0f;

            float blendFactor = (1.0f - accumulatedAlpha) * voxelAlpha;
            accumulatedColor = accumulatedColor + voxelColor * blendFactor;
            accumulatedAlpha = accumulatedAlpha + blendFactor;
        }
        currentDist += fmax(COARSENESSGI, coneWidth * 0.5f);
    }
    return accumulatedColor;
}

/**
 * @brief Calculates the color of the sky for a given view direction.
 * @param dir The normalized view direction.
 * @param sunDir The normalized direction to the sun.
 * @return A float3 representing the sky color.
 */
GPU_FUNC GPU_INLINE half3 sampleSky(const float3 dir, const float3 sunDir)
{
    float sunDot = dot(dir, sunDir);
    if (sunDot > 0.999h) {
        // Convert the constant float3 from cumath.hpp to a glm::vec3 for the return value
        return c_sunColor;
    } else {
        half t = clamp(0.5h * ((half)dir.y + 1.0h), 0.0h, 1.0h);
        return lerp(make_half3(0.2h, 0.4h, 0.8h),   // horizon blue
                    make_half3(0.6h, 0.8h, 1.0h),   // zenith blue
                    t);
    }
}


/**
 * @brief Samples the texture atlas based on voxel position and hit UVs.
 * @param uv The UV coordinates on the face of the hit voxel.
 * @param pos The world position of the hit.
 * @param texObj The CUDA texture object for the texture atlas.
 * @return A float3 representing the albedo color from the texture.
 */
half3 sampleTexture(half2 uv, const float3 pos, TEXTURE_OBJECT texObj);
