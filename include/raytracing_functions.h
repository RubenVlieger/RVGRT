#pragma once
#include "cumath.h" 
#include "renderer/intersections.h"
#include "renderer/ShaderTypes.h"
#include "tables.h"
#include "renderer/hitInfo.h"
class CoarseArray;


// Constants for Indirection
#define BRICK_SIZE 8
#define BRICK_SIZE_SHIFT 3
#define BRICK_MASK 7
#define BRICK_VOLUME 512
#define FLAG_CONSTANT_MAT 0x80000000

constant half3 TINT_GRASS   = half3(0.48h, 0.65h, 0.36h); 
constant half3 TINT_FOLIAGE = half3(0.28h, 0.70h, 0.17h);
constant half3 TINT_NONE    = half3(1.0h, 1.0h, 1.0h);

GPU_FUNC half3 sampleTexture(
    half2 uv, 
    uint8_t matID,
    half3 normal, 
    TEXTURE_OBJECT texObj, 
    float distSq) 
{
    constexpr sampler s(coord::normalized, address::repeat, filter::linear, mip_filter::linear); 
    int face = (normal.y > 0.5h) ? 0 : ((normal.y < -0.5h) ? 1 : 2);
    int texIndex = TEX_STONE; 
    half3 tint = TINT_NONE;

    switch(matID) {
        // --- TERRAIN ---
        case MAT_GRASS:
            if (face == 0) {       texIndex = TEX_GRASS_TOP; tint = TINT_GRASS; } 
            else if (face == 1) {  texIndex = TEX_DIRT; }
            else {                 texIndex = TEX_GRASS_SIDE; }
            break;

        case MAT_DIRT:      texIndex = TEX_DIRT; break;
        case MAT_STONE:     texIndex = TEX_STONE; break;
        case MAT_COBBLE:    texIndex = TEX_COBBLE; break;
        case MAT_BEDROCK:   texIndex = TEX_BEDROCK; break;
        case MAT_SAND:      texIndex = TEX_SAND; break;
        case MAT_GRAVEL:    texIndex = TEX_GRAVEL; break;
        case MAT_CLAY:      texIndex = TEX_CLAY; break;
        case MAT_SOULSAND:  texIndex = TEX_SOULSAND; break;
        case MAT_NETHERRACK:texIndex = TEX_NETHERRACK; break;
        case MAT_GLOWSTONE: texIndex = TEX_GLOWSTONE; break;

        // --- WOOD / CONSTRUCTION ---
        case MAT_PLANKS:    texIndex = TEX_PLANKS; break;
        case MAT_BRICK:     texIndex = TEX_BRICK; break;
        case MAT_MOSSY:     texIndex = TEX_MOSSY; break;
        case MAT_OBSIDIAN:  texIndex = TEX_OBSIDIAN; break;
        
        case MAT_LOG:
            if (face == 0 || face == 1) texIndex = TEX_LOG_TOP; 
            else texIndex = TEX_LOG_SIDE;
            break;

        case MAT_LEAVES:     
            texIndex = TEX_LEAVES; 
            tint = TINT_FOLIAGE; 
            break;

        case MAT_GLASS:     texIndex = TEX_GLASS; break;
        case MAT_WOOL:      texIndex = TEX_WOOL_WHITE; break;
        case MAT_SNOW:      texIndex = TEX_SNOW; break;
        case MAT_ICE:       texIndex = TEX_ICE; break;

        // --- ORES ---
        case MAT_COAL_ORE:  texIndex = TEX_COAL_ORE; break;
        case MAT_IRON_ORE:  texIndex = TEX_IRON_ORE; break;
        case MAT_GOLD_ORE:  texIndex = TEX_GOLD_ORE; break;
        case MAT_DIAM_ORE:  texIndex = TEX_DIAM_ORE; break;

        // --- VALUABLE BLOCKS ---
        case MAT_IRON_BLK:  texIndex = TEX_IRON_BLK; break;
        case MAT_GOLD_BLK:  texIndex = TEX_GOLD_BLK; break;
        case MAT_DIAM_BLK:  texIndex = TEX_DIAM_BLK; break;

        // --- SPECIALS ---
        case MAT_TNT:
            if (face == 0) texIndex = TEX_TNT_TOP;
            else if (face == 1) texIndex = TEX_TNT_BOT;
            else texIndex = TEX_TNT_SIDE;
            break;
            
        case MAT_SANDSTONE:
            if (face == 0) texIndex = TEX_SANDSTONE_TOP; 
            else if (face == 1) texIndex = TEX_SANDSTONE_BOT; 
            else texIndex = TEX_SANDSTONE_SID; 
            break;

        case MAT_PUMPKIN:
            if (face == 0 || face == 1) texIndex = TEX_PUMPKIN_TOP;
            else if (face == 2) texIndex = TEX_PUMPKIN_FACE;
            else texIndex = TEX_PUMPKIN_SIDE;
            break;

        case MAT_CACTUS:
            if (face == 0) texIndex = TEX_CACTUS_TOP;
            else if (face == 1) texIndex = TEX_CACTUS_IN;
            else texIndex = TEX_CACTUS_SIDE;
            break;

        default: 
            return half3(1.0h, 0.0h, 1.0h); 
    }

    float lod = 0.5f * log2(max(distSq, 0.0001f)) - 6.0f;    
    
    half4 t = (half4)texObj.sample(s, float2(uv.x, -uv.y), texIndex, level(lod));
    return t.rgb * tint;
}




GPU_FUNC uint hash3_to_1(int3 p) {
    uint3 u = uint3(p);
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    return u.x ^ u.y ^ u.z;
}
// High-quality, fast Pseudo-Random Number Generator (PCG Hash)
// Essential for path tracing to get "good noise" that denoises well.
GPU_FUNC uint pcg_hash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
// Float random [0, 1]
GPU_FUNC float rand_float(thread uint& seed) {
    seed = pcg_hash(seed);
    return (float)seed / (float)UINT_MAX;
}

// Reconstruct World Position from Depth and Camera info
GPU_FUNC float3 reconstructPos(float depth, float2 uv, constant const CameraData& cam) {
    float2 ndc = uv * 2.0f - 1.0f;
    float3 viewDir = normalize(cam.forward + ndc.x * cam.right + ndc.y * cam.up);
    return cam.position + viewDir * depth;
}

GPU_FUNC half2 reconstructUV(float3 pos, half3 normal) {
    float3 fpos = floor(pos);
    half2 uv;
    if (abs(normal.x) > 0.5h)      uv = half2(pos.y - fpos.y, pos.z - fpos.z);
    else if (abs(normal.y) > 0.5h) uv = half2(pos.x - fpos.x, pos.z - fpos.z);
    else                           uv = half2(pos.x - fpos.x, pos.y - fpos.y);
    return uv;
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


// [cite: 179] Based on existing packing structures (4x4x2)
// GPU_FUNC bool checkBitLocal_Optimized(
//     uint packed,            
//     float3 localEntry,      
//     float3 rayDir,          
//     float3 rayInvDir,       
//     float t_entry_global,   
//     float3 camPos,          
//     thread hitInfo& HI)     
// {
//     if (packed == 0) return false;

//     uint bitXorMask = 0;
//     if (rayDir.x < 0) bitXorMask |= 3u;  // Flip X (0-3)
//     if (rayDir.y < 0) bitXorMask |= 12u; // Flip Y (0-3 shifted by 2)
//     if (rayDir.z < 0) bitXorMask |= 16u; // Flip Z (0-1 shifted by 4)

//     // 3. Positive-Only DDA Setup
//     float3 posDir = abs(rayDir);
//     float3 deltaDist = abs(rayInvDir);
    
//     // In strict positive space, we always step +1. 
//     float3 gridPosF = floor(localEntry);
//     int3 gridPos = int3(gridPosF);
    
//     // Calculate distance to the next *positive* boundary.
//     // If original ray was negative, we essentially measure distance to the 'left' wall 
//     // but treat it as the 'right' wall in mirrored space.
//     float3 sideDist = (sign(rayDir) * (gridPosF - localEntry) + (sign(rayDir) * 0.5f) + 0.5f) * deltaDist;

//     // Mask for bounds checking (Ray exits when it hits index 4 or 2)
//     // We only check positive bounds because we mirrored negative inputs.
//     // 4 in X/Y, 2 in Z.
//     const int3 bounds = int3(4, 4, 2);

//     // 4. Tight Unrolled Traversal
//     // Max path for 4x4x2 is 4+4+2 = 10 steps.
//     // No signs, no direction branching, only comparison.
//     for (int i = 0; i < 10; i++) 
//     {
//         uint bitIndex = (gridPos.z << 4) | (gridPos.y << 2) | gridPos.x;
//         uint finalBit = bitIndex ^ bitXorMask; 

//         // B. Intersection Test
//         if ((packed & (1u << finalBit)) != 0) 
//         {
//             HI.hit = true;

//             float3 t_hit_axis = sideDist - deltaDist;
//             float tLocal = 0.0f;
//             float3 normalLocal = float3(0);

//             // Find max of the entry times (the latest plane we crossed is the entry)
//             if (t_hit_axis.x > t_hit_axis.y && t_hit_axis.x > t_hit_axis.z) {
//                 tLocal = t_hit_axis.x;
//                 normalLocal = float3(-sign(rayDir.x), 0, 0);
//             } else if (t_hit_axis.y > t_hit_axis.z) {
//                 tLocal = t_hit_axis.y;
//                 normalLocal = float3(0, -sign(rayDir.y), 0);
//             } else {
//                 tLocal = t_hit_axis.z;
//                 normalLocal = float3(0, 0, -sign(rayDir.z));
//             }

//             // Handle edge case where ray starts inside solid (tLocal < 0)
//             tLocal = max(0.0f, tLocal);

//             HI.pos = camPos + rayDir * (t_entry_global + tLocal);
//             HI.normal = half3(normalLocal);
//             HI.its += i; 

//             // Fast UV (Planar Projection)
//             float3 fpos = floor(HI.pos);
//             float3 local = HI.pos - fpos;
//             if (abs(HI.normal.x) > 0.5h)      HI.uv = half2(local.y, local.z);
//             else if (abs(HI.normal.y) > 0.5h) HI.uv = half2(local.x, local.z);
//             else                              HI.uv = half2(local.x, local.y);

//             return true;
//         }
//         bool stepX = sideDist.x <= sideDist.y && sideDist.x <= sideDist.z;
//         bool stepY = sideDist.y <  sideDist.x && sideDist.y <= sideDist.z; // strict < to handle equality
//         bool stepZ = !stepX && !stepY;

//         gridPos.x += stepX ? 1 : 0;
//         gridPos.y += stepY ? 1 : 0;
//         gridPos.z += stepZ ? 1 : 0;

//         sideDist.x += stepX ? deltaDist.x : 0.0f;
//         sideDist.y += stepY ? deltaDist.y : 0.0f;
//         sideDist.z += stepZ ? deltaDist.z : 0.0f;

//         if (gridPos.x >= bounds.x || gridPos.y >= bounds.y || gridPos.z >= bounds.z) break;
//     }

//     return false;
// }


GPU_FUNC bool checkBitLocal_Optimized(
    uint packed,            // The geometry data (32 bits)
    float3 localEntry,      // Entry position relative to sub-chunk (0..4, 0..4, 0..2)
    float3 rayDir,          // Ray Direction
    float3 rayInvDir,       // 1.0 / Ray Direction (Precomputed)
    int3 stepSign,          // Ray Step Sign (-1 or 1)
    float t_entry_global,   // Global T where we hit this sub-chunk boundary
    float3 camPos,          // Origin for reconstruction
    thread hitInfo& HI)     // Output ref
{
    int3 mapPos = make_int3(
        clamp((int)floor(localEntry.x), 0, 3),
        clamp((int)floor(localEntry.y), 0, 3),
        clamp((int)floor(localEntry.z), 0, 1)
    );

    int currentBitIndex = (mapPos.z << 4) | (mapPos.y << 2) | mapPos.x;

    const int3 stepStride = make_int3(stepSign.x, stepSign.y * 4, stepSign.z * 16);

    float3 deltaDist = abs(rayInvDir);
    
    float3 originOffset = float3(mapPos) - localEntry;
    
    float3 sideDist;
    sideDist.x = (stepSign.x > 0? (originOffset.x + 1.0f) : -originOffset.x) * deltaDist.x;
    sideDist.y = (stepSign.y > 0? (originOffset.y + 1.0f) : -originOffset.y) * deltaDist.y;
    sideDist.z = (stepSign.z > 0? (originOffset.z + 1.0f) : -originOffset.z) * deltaDist.z;

    float tLocalHit = 0.0f; 
    int lastAxis = -1;
    #pragma unroll 10
    for (int i = 0; i < 10; i++) {
        
        if ((packed & (1u << currentBitIndex))!= 0) {
            HI.hit = true;

            half3 nx = make_half3((half)(-stepSign.x), 0, 0);
            half3 ny = make_half3(0, (half)(-stepSign.y), 0);
            half3 nz = make_half3(0, 0, (half)(-stepSign.z));
            
            half3 nDefault = make_half3(0, 1, 0);

            HI.normal = (lastAxis == 0)? nx : ((lastAxis == 1)? ny : ((lastAxis == 2)? nz : nDefault));

            HI.pos = camPos + rayDir * (t_entry_global + tLocalHit);
            HI.its += i;

            float3 fpos = floor(HI.pos);
            if (abs(HI.normal.x) > 0.5h)      HI.uv = half2(HI.pos.y - fpos.y, HI.pos.z - fpos.z);
            else if (abs(HI.normal.y) > 0.5h) HI.uv = half2(HI.pos.x - fpos.x, HI.pos.z - fpos.z);
            else                              HI.uv = half2(HI.pos.x - fpos.x, HI.pos.y - fpos.y);
            
            return true;
        }

        bool xMin = (sideDist.x <= sideDist.y) && (sideDist.x <= sideDist.z);
        bool yMin = (sideDist.y <= sideDist.z) &&!xMin;

        tLocalHit = xMin? sideDist.x : (yMin? sideDist.y : sideDist.z);

        if (xMin) sideDist.x += deltaDist.x;
        else if (yMin) sideDist.y += deltaDist.y;
        else sideDist.z += deltaDist.z;

        if (xMin) {
            mapPos.x += stepSign.x;
            currentBitIndex += stepStride.x;
            lastAxis = 0;
        } else if (yMin) {
            mapPos.y += stepSign.y;
            currentBitIndex += stepStride.y;
            lastAxis = 1;
        } else {
            mapPos.z += stepSign.z;
            currentBitIndex += stepStride.z;
            lastAxis = 2;
        }
        if ((mapPos.x & ~3) | (mapPos.y & ~3) | (mapPos.z & ~1)) break;
    }

    return false;
}
/**
 * @brief Calculates the color of the sky for a given view direction.
 * @param dir The normalized view direction.
 * @param sunDir The normalized direction to the sun.
 * @return A float3 representing the sky color.
 */
inline half3 sampleSky(const float3 dir, const float3 sunDir)
{
    float sunDot = dot(dir, sunDir);
    
    // Sun Disk (Sharp)
    if (sunDot > 0.999h) {
        return c_sunColor * 2.0h; // Super bright sun disk
    } 
    
    // Gradient: Deep Blue at top, lighter blue at horizon
    float y = clamp(dir.y, 0.0f, 1.0f);
    
    // Richer Blues for vibrant look
    half3 zenith = half3(0.1h, 0.4h, 0.8h);  // Deep blue top
    half3 horizon = half3(0.4h, 0.6h, 0.9h); // Cyan/White horizon
    
    half3 skyColor = lerp(horizon, zenith, half(pow(y, 0.7f)));
    
    return skyColor;
}


/**
 * @brief Samples the texture atlas based on voxel position and hit UVs.
 * @param uv The UV coordinates on the face of the hit voxel.
 * @param pos The world position of the hit.
 * @param texObj The CUDA texture object for the texture atlas.
 * @return A float3 representing the albedo color from the texture.
 */
half3 sampleTexture(half2 uv, const float3 pos, TEXTURE_OBJECT texObj, float depth);



