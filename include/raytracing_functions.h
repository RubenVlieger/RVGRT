#include "cumath.h" 
#include "CoarseArray.h"
class CoarseArray;

struct hitInfo
{
    float3 pos;
    float3 normal;
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
GPU_FUNC GPU_INLINE bool IsSolid(const int3 p, DEVICE_PTR(const uint32_t*) RESTRICT bits)
{
    uint64_t index = toIndex(p.x, p.y, p.z);
    return ((bits[index >> 5] >> (index & 31)) & 1);
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
GPU_FUNC GPU_INLINE float3 clamp3(const float3 a, const float3 b, const float3 c)
{
    float3 d = make_float3(a.x < b.x ? b.x : a.x,
                       a.y < b.y ? b.y : a.y,
                       a.z < b.z ? b.z : a.z);
    return make_float3(d.x > c.x ? c.x : d.x,
                     d.y > c.y ? c.y : d.y,
                     d.z > c.z ? c.z : d.z);
}

/**
 * @brief Samples the Coarse Signed Distance Field (CSDF) at a floating-point world position.
 * @param pos The world position to sample from.
 * @param csdf Pointer to the CSDF data on the GPU.
 * @return The approximate distance to the nearest surface in coarse grid units.
 */
GPU_FUNC GPU_INLINE float getDistance(const float3 pos, DEVICE_PTR(const uint8_t*) RESTRICT csdf)
{
    float3 tc = floor3(pos * (1.0f / (float)COARSENESSSDF));
    int3 c = make_int3(tc.x, tc.y, tc.z);

    c = clamp3(c, make_int3(0), make_int3(SDF_SIZEX - 1, SDF_SIZEY - 1, SDF_SIZEZ - 1));

    int cidx = c.z * SDF_SIZEX * SDF_SIZEY + c.y * SDF_SIZEX + c.x;
    return (float)csdf[cidx];
}

/**
 * @brief Samples the CSDF at an integer voxel coordinate.
 * @param pos The integer voxel coordinates.
 * @param csdf Pointer to the CSDF data on the GPU.
 * @return The approximate distance from the coarse cell containing the voxel.
 */
GPU_FUNC GPU_INLINE float getDistance(const int3 pos, DEVICE_PTR(const uint8_t*) RESTRICT csdf)
{
    int3 c = make_int3(pos.x / COARSENESSSDF, pos.y / COARSENESSSDF, pos.z / COARSENESSSDF);

    c = clamp3(c, make_int3(0), make_int3(SDF_SIZEX - 1, SDF_SIZEY - 1, SDF_SIZEZ - 1));
    int cidx = c.z * SDF_SIZEX * SDF_SIZEY + c.y * SDF_SIZEX + c.x;
    return csdf[cidx];
}

/**
 * @brief Marches a ray using the CSDF to quickly find a point near a surface.
 * @param pos The starting position of the ray.
 * @param dir The normalized direction of the ray.
 * @param csdf Pointer to the CSDF data on the GPU.
 * @return A float3 point on the ray that is close to a solid surface.
 */
GPU_FUNC float3 approximateCSDF(float3 pos, const float3 dir, DEVICE_PTR(const uint8_t*) RESTRICT csdf)
{
    for(int i = 0; i < 100; ++i) {
        if ((pos.x < 0) | (pos.y < 0) | (pos.z < 0) || 
            (pos.x >= SIZEX) | (pos.y >= SIZEY) | (pos.z >= SIZEZ)) {
            return make_float3(-100.f);
        }
        float dist = getDistance(pos, csdf);
        if (dist <= 1.0f) return pos;
        pos = pos + dir * dist;
    }
    return pos;
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
GPU_FUNC hitInfo trace(float3 camPos, const float3 camDir, half distance,
                       DEVICE_PTR(const uint32_t*) RESTRICT bits, 
                       DEVICE_PTR(const uint8_t*) RESTRICT csdf)
{
    hitInfo HI;
    HI.hit = false;
    HI.its = 0;
    HI.pos = make_float3(-500.f);

    float3 currentPos = camPos + camDir * (float)distance;

    float3 deltaDist = abs3(make_float3(1.0f) / camDir);
    int3 step = make_int3(camDir.x > 0, camDir.y > 0, camDir.z > 0);

    bool jumped = false;
    for (int majorIteration = 0; majorIteration < 5; majorIteration++)
    {
        HI.its++;
        currentPos = approximateCSDF(currentPos, camDir, csdf);
        
        int3 ipos = to_int3(floor3(currentPos));
        float3 tMax = (make_float3(ipos) - currentPos + (make_float3(step) * 0.5f) + make_float3(0.5f)) * deltaDist;
        
        uint8_t mask = -128;
        for (int i = 0; i < 200; i++) {
            HI.its++;
            if ((i & 7) == 7) {
                uint8_t dist = getDistance(ipos, csdf);
                if (dist > 2) {
                    currentPos = currentPos + camDir * ((float)dist * COARSENESSSDF);
                    jumped = true;
                    break;
                }
            }
            
            if (((ipos.x < 0) | (ipos.y < 0) | (ipos.z < 0) || 
                (ipos.x >= SIZEX) | (ipos.y >= SIZEY) | (ipos.z >= SIZEZ))) {
                return HI; 
            }
            
            if (IsSolid(ipos, bits)) {
                HI.hit = true;
                if (mask == 0) {
                    HI.normal = make_float3(-step.x, 0.f, 0.f);
                    HI.pos = currentPos + camDir * (tMax.x - deltaDist.x);
                    HI.uv = make_half2(HI.pos.y - ipos.y, HI.pos.z - ipos.z);
                    if(step.x == -1) HI.uv.y = (half)1.0f - HI.uv.y;
                } else if (mask == 1) {
                    HI.normal = make_float3(0, -step.y, 0);
                    HI.pos = currentPos + camDir * (tMax.y - deltaDist.y);
                    HI.uv = make_half2(HI.pos.x - ipos.x, HI.pos.z - ipos.z);
                } else { // mask == 2
                    HI.normal = make_float3(0, 0, -step.z);
                    HI.pos = currentPos + camDir * (tMax.z - deltaDist.z);
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


/**
 * @brief Traces a cone through the GI data grid to gather indirect illumination.
 * @param pos The starting point of the cone (a surface point).
 * @param dir The central direction of the cone.
 * @param GIdata Pointer to the 3D grid of GI voxel data.
 * @param csdf Pointer to the CSDF for occlusion checks.
 * @return A float3 color representing the accumulated indirect light.
 */
GPU_FUNC float3 traceCone(float3 pos, const float3 dir, 
                          DEVICE_PTR(const uint32_t*) RESTRICT GIdata,
                          DEVICE_PTR(const uint8_t*) RESTRICT csdf)
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
        
        if (!((g.x < 0) | (g.y < 0) | (g.z < 0) || 
              (g.x >= GI_SIZEX) | (g.y >= GI_SIZEY) | (g.z >= GI_SIZEZ))) 
        {       
            uint64_t gidx = (uint64_t)g.z * GI_SIZEX * GI_SIZEY + (uint64_t)g.y * GI_SIZEX + g.x;
            uint32_t giSample = GIdata[gidx];
            
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
GPU_FUNC GPU_INLINE float3 sampleSky(const float3 dir, const float3 sunDir)
{
    float sunDot = dot(dir, sunDir);
    if (sunDot > 0.999f) {
        // Convert the constant float3 from cumath.hpp to a glm::vec3 for the return value
        return c_sunColor;
    } else {
        float t = clamp(0.5f * (dir.y + 1.0f), 0.0f, 1.0f);
        return lerp(make_float3(0.2f, 0.4f, 0.8f),   // horizon blue
                    make_float3(0.6f, 0.8f, 1.0f),   // zenith blue
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
float3 sampleTexture(half2 uv, const float3 pos, TEXTURE_OBJECT texObj);
