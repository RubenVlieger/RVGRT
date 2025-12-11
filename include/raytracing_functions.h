#include "cumath.h" 

#include "renderer/ShaderTypes.h"
class CoarseArray;

struct hitInfo
{
    float3 pos;
    half3 normal;
    half2 uv; 
    bool hit;
    int its;
};

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
    float3 pos, 
    half3 normal, 
    texture2d_array<float, access::sample> texObj, 
    float distSq, 
    texture3d<uint, access::read> indirection, 
    device uchar* matPool) 
{
    // 1. Get Material ID
    uint3 coarsePos = uint3(pos / 8.0f);
    
    // Bounds check
    if(coarsePos.x >= indirection.get_width() || coarsePos.y >= indirection.get_height() || coarsePos.z >= indirection.get_depth()) 
        return half3(0,0,0);

    uint cellData = indirection.read(coarsePos).r;
    uint8_t matID = MAT_STONE; // Default

    if (cellData == FLAG_SOLID_GENERIC) {
        matID = MAT_BEDROCK; // Or generic solid material
    } 
    else if (cellData >= IND_OFFSET) {
        uint brickIdx = cellData - IND_OFFSET;

        uint3 localPos = uint3(pos) & 7;
        
        // Linear Voxel Index: z*64 + y*8 + x
        uint localLinear = localPos.x + (localPos.y * 8) + (localPos.z * 64);
        
        // Final Address
        uint finalAddr = (brickIdx * 512) + localLinear;
        
        // 3. Read Buffer (Single Load)
        matID = matPool[finalAddr];
    }
    constexpr sampler s(coord::normalized, address::repeat, filter::linear, mip_filter::linear); 
    int face = (normal.y > 0.5h) ? 0 : ((normal.y < -0.5h) ? 1 : 2);
    int texIndex = TEX_STONE; 
    half3 tint = TINT_NONE;

    switch(matID) {
        // --- TERRAIN ---
        case MAT_GRASS:
            if (face == 0) {       texIndex = TEX_GRASS_TOP; tint = TINT_GRASS; } 
            else if (face == 1) {  texIndex = TEX_DIRT; }
            else {                 texIndex = TEX_GRASS_SIDE; } // Side has overlay, hard to tint correctly in single pass, usually looks OK without tint or specific biome logic
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
            else if (face == 2) texIndex = TEX_PUMPKIN_FACE; // Assuming 'side' implies front here for simplicity
            else texIndex = TEX_PUMPKIN_SIDE;
            break;

        case MAT_CACTUS:
            if (face == 0) texIndex = TEX_CACTUS_TOP;
            else if (face == 1) texIndex = TEX_CACTUS_IN;
            else texIndex = TEX_CACTUS_SIDE;
            break;

        default: 
            // Debug Pink
            return half3(1.0h, 0.0h, 1.0h); 
    }

    float lod = 0.5f * log2(distSq) - 6.0f;
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

GPU_FUNC bool checkBitLocal(uint packed, int lx, int ly, int lz) {
    // lx (0-3), ly (0-3), lz (0-1)
    // Flatten: z*16 + y*4 + x
    int bitIndex = lx + (ly << 2) + (lz << 4);
    return (packed & (1u << bitIndex)) != 0;
}

// GPU_FUNC bool checkBitLocal_Optimized(
//     uint packed, float3 localEntry, float3 rayInvDir, int3 stepSign) 
// {
//     int3 mapPos = make_int3(
//         clamp((int)floor(localEntry.x), 0, 3),
//         clamp((int)floor(localEntry.y), 0, 3),
//         clamp((int)floor(localEntry.z), 0, 1)
//     );

//     int currentBitIndex = (mapPos.z << 4) | (mapPos.y << 2) | mapPos.x;

//     const int3 stepStride = make_int3(stepSign.x, stepSign.y * 4, stepSign.z * 16);

//     float3 deltaDist = abs(rayInvDir);
    
//     float3 originOffset = float3(mapPos) - localEntry;
    
//     float3 sideDist;
//     sideDist.x = (stepSign.x > 0? (originOffset.x + 1.0f) : -originOffset.x) * deltaDist.x;
//     sideDist.y = (stepSign.y > 0? (originOffset.y + 1.0f) : -originOffset.y) * deltaDist.y;
//     sideDist.z = (stepSign.z > 0? (originOffset.z + 1.0f) : -originOffset.z) * deltaDist.z;

//     #pragma unroll 10
//     for (int i = 0; i < 10; i++) {
//         if ((packed & (1u << currentBitIndex))!= 0) return true;

//         bool xMin = (sideDist.x <= sideDist.y) && (sideDist.x <= sideDist.z);
//         bool yMin = (sideDist.y <= sideDist.z) &&!xMin;

//         if (xMin) { sideDist.x += deltaDist.x; mapPos.x += stepSign.x; currentBitIndex += stepStride.x; }
//         else if (yMin) { sideDist.y += deltaDist.y; mapPos.y += stepSign.y; currentBitIndex += stepStride.y; }
//         else { sideDist.z += deltaDist.z; mapPos.z += stepSign.z; currentBitIndex += stepStride.z; }

//         if ((mapPos.x & ~3) | (mapPos.y & ~3) | (mapPos.z & ~1)) break;
//     }
//     return false;
// }


GPU_FUNC hitInfo trace(float3 camPos, 
                       float3 camDir, 
                       texture3d<uint, access::read> indirection,
                       device uint* geoPool)
{
    hitInfo HI;
    HI.hit = false;
    HI.its = 0;
    HI.pos = make_float3(0.0f);
    HI.normal = make_half3(0.0h);

    // --- Init DDA ---
    float t_current = 0.0f;
    float3 deltaDist = abs(1.0f / (camDir + 1e-10f));
    float3 rayInvDir = 1.0f / (camDir + 1e-5f);
    int3 step = (int3)(sign(camDir));
    
    // Calculate initial grid state
    float3 rayStart = camPos / 8.0f;
    int3 mapPos = int3(floor(rayStart));
    float3 sideDist = (sign(camDir) * (float3(mapPos) - rayStart) + (sign(camDir) * 0.5f) + 0.5f) * deltaDist;

    // We use a safe max loop count. 
    // Since we skip empty space, 128 iterations covers HUGE distances.
    for (int i = 0; i < 512; i++) {
        HI.its++;

        // 1. Bounds Check
        if (mapPos.x < 0 || mapPos.y < 0 || mapPos.z < 0 ||
            mapPos.x >= indirection.get_width() || 
            mapPos.y >= indirection.get_height() || 
            mapPos.z >= indirection.get_depth()) break;

        // 2. Read Indirection (SDF + Type)
        uint val = indirection.read(uint3(mapPos)).r;
        uint distByte = val >> 24;      // Top 8 bits: Macro-SDF
        uint cellData = val & 0x00FFFFFF; // Bottom 24 bits: Type/Index

        // 3. LOGIC: SDF SKIP
        // If we are far from geometry, jump!
        if (distByte > 1) {
            // Convert Brick Distance to World Units (minus safety margin)
            float skipDist = (float(distByte)) * 8.0f;
            
            t_current += skipDist;
            
            // Re-project ray to new position
            float3 newRayPos = (camPos + camDir * t_current) / 8.0f;
            mapPos = int3(floor(newRayPos));
            
            // Recalculate DDA state (sideDist) for the new block
            // This effectively "resets" the DDA further down the line
            sideDist = (sign(camDir) * (float3(mapPos) - newRayPos) + (sign(camDir) * 0.5f) + 0.5f) * deltaDist;
            
            continue; // Jump to next iteration
        }

        // --- Prepare for Local Intersection (Solid or Mixed) ---
        // We calculate entry time relative to the *current* DDA step
        // tCand represents "How far did we travel in Brick Units to get to this boundary?"
        float3 tCand = sideDist - deltaDist;
        int axis = 0;
        if (tCand.x >= tCand.y && tCand.x >= tCand.z) axis = 0;
        else if (tCand.y >= tCand.z) axis = 1;
        else axis = 2;
        
        float tCoarseLocal = fmax(0.0f, tCand[axis]); // Local T within DDA
        float tWorldTotal = t_current + tCoarseLocal * 8.0f; // Absolute World T

        // 4. LOGIC: SOLID HIT
        if (cellData == FLAG_SOLID_GENERIC) {
            HI.hit = true;
            if (axis == 0) HI.normal = make_half3(-step.x, 0, 0);
            else if (axis == 1) HI.normal = make_half3(0, -step.y, 0);
            else                HI.normal = make_half3(0, 0, -step.z);
            
            HI.pos = camPos + camDir * tWorldTotal;
            HI.uv = reconstructUV(HI.pos, HI.normal); 
            return HI;
        }

        // 5. LOGIC: MIXED BRICK (Fine Traversal)
        else if (cellData >= IND_OFFSET) {
            uint brickIdx = cellData - IND_OFFSET;
            uint geoBase = brickIdx * 16;

            // -----------------------------------------------------
            // OPTIMIZED INNER TRAVERSAL (Sub-Chunk DDA)
            // -----------------------------------------------------
            // We are traversing a 2x2x4 grid of "Sub-Chunks".
            // Sub-chunk size: 4.0, 4.0, 2.0
            
            float3 brickOrigin = float3(mapPos) * 8.0f;
            float3 rayP = camPos + camDir * t_current; // Entry point in world
            float3 localP = rayP - brickOrigin; // 0..8

            // DDA Setup for the 2x2x4 grid
            // Grid dimensions
            const float3 subSize = float3(4.0f, 4.0f, 2.0f);
            
            // Current Sub-Chunk Index (0..1, 0..1, 0..3)
            int3 subMapPos = int3(floor(localP / subSize));
            subMapPos = clamp(subMapPos, int3(0), int3(1, 1, 3)); // Safety clamp

            // Delta Dist for Sub-Chunks
            float3 subDeltaDist = abs(subSize / camDir);
            float3 subSideDist;
            
            // Calculate initial sideDist relative to sub-grid
            float3 subRayStart = localP / subSize;
            subSideDist = (sign(camDir) * (float3(subMapPos) - subRayStart) + (sign(camDir) * 0.5f) + 0.5f) * subDeltaDist;
            
            // Traverse the 16 sub-chunks (Max steps roughly 6-8)
            for (int k=0; k<8; k++) {
                // Bounds check local 2x2x4 grid
                if (subMapPos.x < 0 || subMapPos.x > 1 || 
                    subMapPos.y < 0 || subMapPos.y > 1 || 
                    subMapPos.z < 0 || subMapPos.z > 3) break;

                // 1. READ BUFFER (Linear Index)
                // Linearize: x + y*2 + z*4
                uint subIdx = subMapPos.x + (subMapPos.y * 2) + (subMapPos.z * 4);
                uint packedGeo = geoPool[geoBase + subIdx];

                // 2. CHECK INTERSECTION
                // We assume checkBitLocal calculates precise intersection if packed != 0
                if (packedGeo != 0) {
                    // Refine intersection within this 4x4x2 block
                    // We need to pass the offset of this sub-chunk
                    float3 subChunkOrigin = brickOrigin + float3(subMapPos.x*4, subMapPos.y*4, subMapPos.z*2);
                    float3 localEntryPos = (camPos + camDir * (t_current + 0.001f)) - subChunkOrigin;

                    
                    // Call your existing bit-check logic logic adapted for specific sub-block
                    // If hit, return HI
                    // This creates the "Hit" logic without looping 24 times blindly
                    if (checkBitLocal_Optimized(packedGeo, localEntryPos, camDir, rayInvDir, step, t_current, camPos, HI)) {
                        return HI;
                    }
                }

                // 3. STEP SUB-CHUNK DDA
                if (subSideDist.x < subSideDist.y) {
                    if (subSideDist.x < subSideDist.z) {
                        subSideDist.x += subDeltaDist.x; subMapPos.x += step.x;
                    } else {
                        subSideDist.z += subDeltaDist.z; subMapPos.z += step.z;
                    }
                } else {
                    if (subSideDist.y < subSideDist.z) {
                        subSideDist.y += subDeltaDist.y; subMapPos.y += step.y;
                    } else {
                        subSideDist.z += subDeltaDist.z; subMapPos.z += step.z;
                    }
                }
            }
        }
        if (sideDist.x < sideDist.y) {
            if (sideDist.x < sideDist.z) {
                sideDist.x += deltaDist.x; mapPos.x += step.x;
            } else {
                sideDist.z += deltaDist.z; mapPos.z += step.z;
            }
        } else {
            if (sideDist.y < sideDist.z) {
                sideDist.y += deltaDist.y; mapPos.y += step.y;
            } else {
                sideDist.z += deltaDist.z; mapPos.z += step.z;
            }
        }
    }
    
    return HI;
}


// GPU_FUNC hitInfo trace(float3 camPos, 
//                        float3 camDir, 
//                        texture3d<uint, access::read> indirection,
//                        texture3d<uint, access::read> geoPool,
//                        texture3d<uint, access::read> matPool,
//                        uint3 atlasDim)
// {
//     hitInfo HI;
//     HI.hit = false;
//     HI.its = 0;
//     HI.pos = make_float3(0.0f);
//     HI.normal = make_half3(0.0h);
    
//     // --- Phase 1: Coarse DDA Setup (Brick Space) ---
//     float3 rayStart = camPos / (float)BRICK_SIZE;
//     float3 deltaDist = abs(1.0f / (camDir + 1e-10f)); 

//     int3 step = make_int3(
//         camDir.x > 0 ? 1 : -1,
//         camDir.y > 0 ? 1 : -1,
//         camDir.z > 0 ? 1 : -1
//         );
//     int3 mapPos = int3(floor(rayStart));
//     float3 sideDist = (sign(camDir) * (float3(mapPos) - rayStart) + (sign(camDir) * 0.5f) + 0.5f) * deltaDist;
    
//     for (int i = 0; i < 256; i++) {
//         HI.its++;
        
//         if (mapPos.x < 0 || mapPos.y < 0 || mapPos.z < 0 ||
//             mapPos.x >= indirection.get_width() || 
//             mapPos.y >= indirection.get_height() || 
//             mapPos.z >= indirection.get_depth()) break;

//         uint cellData = indirection.read(uint3(mapPos)).r;
        
//         // --- Calculate Coarse Entry ---
//         // Determine which face we entered the brick through
//         float3 tCand = sideDist - deltaDist;
//         int coarseAxis = 0;
//         if (tCand.x >= tCand.y && tCand.x >= tCand.z) coarseAxis = 0;
//         else if (tCand.y >= tCand.z) coarseAxis = 1;
//         else coarseAxis = 2;
        
//         float tCoarse = fmax(0.0f, tCand[coarseAxis]);

//         if (cellData == FLAG_SOLID_GENERIC) {
//             HI.hit = true;
//             if (coarseAxis == 0) HI.normal = make_half3(-step.x, 0, 0);
//             else if (coarseAxis == 1) HI.normal = make_half3(0, -step.y, 0);
//             else                      HI.normal = make_half3(0, 0, -step.z);
//             HI.pos = camPos + camDir * (tCoarse * (float)BRICK_SIZE);
//             HI.uv = reconstructUV(HI.pos, HI.normal); 
//             return HI;
//         } 
//         else if (cellData >= IND_OFFSET) {
//             uint atlasIdx = cellData - IND_OFFSET;
//             uint area = atlasDim.x * atlasDim.y;
//             uint bz = atlasIdx / area;
//             uint rem = atlasIdx % area;
//             uint by = rem / atlasDim.x;
//             uint bx = rem % atlasDim.x;

//             // 1. Precise Local Start
//             // Scale coarse entry time to world units
//             float3 worldEntry = camPos + camDir * (tCoarse * (float)BRICK_SIZE + 0.0001f);
//             float3 brickOrigin = float3(mapPos) * (float)BRICK_SIZE;
//             float3 localPos = worldEntry - brickOrigin;

//             // Snap the entry axis coordinate to exact boundary (0 or 8)
//             if (coarseAxis == 0) localPos.x = (step.x > 0) ? 0.0f : 8.0f;
//             if (coarseAxis == 1) localPos.y = (step.y > 0) ? 0.0f : 8.0f;
//             if (coarseAxis == 2) localPos.z = (step.z > 0) ? 0.0f : 8.0f;

//             // 2. Setup Local DDA
//             int3 localMapPos = int3(floor(localPos));
            
//             // Force entry index to be valid
//             if (coarseAxis == 0) localMapPos.x = (step.x > 0) ? 0 : 7;
//             if (coarseAxis == 1) localMapPos.y = (step.y > 0) ? 0 : 7;
//             if (coarseAxis == 2) localMapPos.z = (step.z > 0) ? 0 : 7;
            
//             localMapPos = clamp(localMapPos, int3(0), int3(7));
            
//             float3 localSideDist = (sign(camDir) * (float3(localMapPos) - localPos) + (sign(camDir) * 0.5f) + 0.5f) * deltaDist;
            
//             // Track the axis we used to enter the CURRENT voxel
//             int lastAxis = coarseAxis; 

//             for(int k=0; k<24; k++) {
//                 if(localMapPos.x < 0 || localMapPos.y < 0 || localMapPos.z < 0 ||
//                    localMapPos.x > 7 || localMapPos.y > 7 || localMapPos.z > 7) break; 
                   
//                 uint3 subBlock = uint3(localMapPos.x >> 2, localMapPos.y >> 2, localMapPos.z >> 1);
//                 uint3 geoCoord = uint3(bx * 2, by * 2, bz * 4) + subBlock;
//                 uint packedGeo = geoPool.read(geoCoord).r;
                
//                 if(checkBitLocal(packedGeo, localMapPos.x & 3, localMapPos.y & 3, localMapPos.z & 1)) {
//                     HI.hit = true;
                    
//                     // CORRECTED LOGIC: Use lastAxis (Entry) not min(sideDist) (Exit)
//                     float tLocal = 0.0f;
                    
//                     if (lastAxis == 0) {
//                         HI.normal = make_half3(-step.x, 0, 0);
//                         tLocal = localSideDist.x - deltaDist.x;
//                     } else if (lastAxis == 1) {
//                         HI.normal = make_half3(0, -step.y, 0);
//                         tLocal = localSideDist.y - deltaDist.y;
//                     } else {
//                         HI.normal = make_half3(0, 0, -step.z);
//                         tLocal = localSideDist.z - deltaDist.z;
//                     }
//                     float tTotal = tCoarse * 8.0f + tLocal;
                    
//                     HI.pos = camPos + camDir * tTotal;
//                     HI.uv = reconstructUV(HI.pos, HI.normal);
//                     return HI;
//                 }
                
//                 if (localSideDist.x < localSideDist.y) {
//                     if (localSideDist.x < localSideDist.z) {
//                         localSideDist.x += deltaDist.x; localMapPos.x += step.x; 
//                         lastAxis = 0;
//                     } else {
//                         localSideDist.z += deltaDist.z; localMapPos.z += step.z; 
//                         lastAxis = 2;
//                     }
//                 } else {
//                     if (localSideDist.y < localSideDist.z) {
//                         localSideDist.y += deltaDist.y; localMapPos.y += step.y; 
//                         lastAxis = 1;
//                     } else {
//                         localSideDist.z += deltaDist.z; localMapPos.z += step.z; 
//                         lastAxis = 2;
//                     }
//                 }
//             }
//         }

//         // Coarse Step
//         if (sideDist.x < sideDist.y) {
//             if (sideDist.x < sideDist.z) {
//                 sideDist.x += deltaDist.x; mapPos.x += step.x;
//             } else {
//                 sideDist.z += deltaDist.z; mapPos.z += step.z;
//             }
//         } else {
//             if (sideDist.y < sideDist.z) {
//                 sideDist.y += deltaDist.y; mapPos.y += step.y;
//             } else {
//                 sideDist.z += deltaDist.z; mapPos.z += step.z;
//             }
//         }
//     }
//     return HI;
// }


// Returns true if blocked
// -----------------------------------------------------------------------------
// HELPER: Optimized Shadow Bit-Traverser
// -----------------------------------------------------------------------------
// Checks intersection against a 4x4x2 sub-chunk stored in a single uint.
// Returns TRUE immediately if any solid bit is hit.
// -----------------------------------------------------------------------------
GPU_FUNC bool checkBitLocal_Shadow(uint packed, float3 localEntry, float3 rayDir, float3 rayInvDir, int3 step) 
{
    // 1. Setup Local DDA for 1x1x1 voxels
    int3 mapPos = int3(floor(localEntry));
    
    // Clamp to valid range (0-3, 0-3, 0-1) to handle floating point edge cases
    mapPos.x = clamp(mapPos.x, 0, 3);
    mapPos.y = clamp(mapPos.y, 0, 3);
    mapPos.z = clamp(mapPos.z, 0, 1);

    // Delta is constant for the ray
    float3 deltaDist = abs(rayInvDir);
    
    // Calculate initial sideDist
    float3 sideDist = (sign(rayDir) * (float3(mapPos) - localEntry) + (sign(rayDir) * 0.5f) + 0.5f) * deltaDist;
    
    // 2. Register-based Traversal Loop (Unrolled)
    // Max path length in 4x4x2 grid is small (< 12)
    for (int i = 0; i < 12; i++) {
        
        // Bit Index: (z * 16) + (y * 4) + x
        // Optimized: (z << 4) | (y << 2) | x
        int bitIndex = (mapPos.z << 4) | (mapPos.y << 2) | mapPos.x;
        
        // Check Bit
        if ((packed & (1u << bitIndex)) != 0) {
            return true; // Hit!
        }

        // Step DDA
        if (sideDist.x < sideDist.y) {
            if (sideDist.x < sideDist.z) {
                sideDist.x += deltaDist.x;
                mapPos.x += step.x;
            } else {
                sideDist.z += deltaDist.z;
                mapPos.z += step.z;
            }
        } else {
            if (sideDist.y < sideDist.z) {
                sideDist.y += deltaDist.y;
                mapPos.y += step.y;
            } else {
                sideDist.z += deltaDist.z;
                mapPos.z += step.z;
            }
        }

        // Exit check
        if (mapPos.x < 0 || mapPos.x > 3 || 
            mapPos.y < 0 || mapPos.y > 3 || 
            mapPos.z < 0 || mapPos.z > 1) break;
    }

    return false;
}
// include/raytracing_functions.h

GPU_FUNC bool traceShadowCoarse(float3 startPos, 
                                float3 lightDir, // Max dist in World Units
                                texture3d<uint, access::read> indirection)
{
    // 1. Setup DDA (Standard)
    float3 rayInvDir = 1.0f / (lightDir + 1e-5f);
    float3 deltaDist = abs(rayInvDir); 
    int3 step = int3(sign(lightDir));

    float3 rayStart = startPos / 8.0f; // To Brick Space
    int3 mapPos = int3(floor(rayStart));
    
    float3 sideDist = (sign(lightDir) * (float3(mapPos) - rayStart) + (sign(lightDir) * 0.5f) + 0.5f) * deltaDist;

    
    // Track current T (Euclidean distance along ray)
    float currentT = 0.0f;

    for (int i = 0; i < 256; i++) {
        // 2. Bounds Check
        if (mapPos.x < 0 || mapPos.y < 0 || mapPos.z < 0 ||
            mapPos.x >= indirection.get_width() || 
            mapPos.y >= indirection.get_height() || 
            mapPos.z >= indirection.get_depth()) return false;

        // 4. Fetch
        uint val = indirection.read(uint3(mapPos)).r;
        uint distByte = val >> 24;      
        uint cellData = val & 0x00FFFFFF;

        // 5. Logic: SDF Skip
        if (distByte > 1) {
            float skipDist = float(distByte) - 1.0f; 
            
            // Advance T
            currentT += skipDist;
            
            float3 p = rayStart + lightDir * currentT;
            mapPos = int3(floor(p));
            
            sideDist = (sign(lightDir) * (float3(mapPos) - p) + (sign(lightDir) * 0.5f) + 0.5f) * deltaDist;

            sideDist += currentT;
            
            continue;
        }

        // 6. Hit Check (Conservative)
        if (cellData != 0) return true;

        // 7. Step DDA
        if (sideDist.x < sideDist.y) {
            if (sideDist.x < sideDist.z) {
                currentT = sideDist.x; // We move TO sideDist.x
                sideDist.x += deltaDist.x; 
                mapPos.x += step.x;
            } else {
                currentT = sideDist.z;
                sideDist.z += deltaDist.z; 
                mapPos.z += step.z;
            }
        } else {
            if (sideDist.y < sideDist.z) {
                currentT = sideDist.y;
                sideDist.y += deltaDist.y; 
                mapPos.y += step.y;
            } else {
                currentT = sideDist.z;
                sideDist.z += deltaDist.z; 
                mapPos.z += step.z;
            }
        }
    }
    
    return false;
}



// -----------------------------------------------------------------------------
// MAIN: TraceShadow
// -----------------------------------------------------------------------------
GPU_FUNC bool traceShadow(float3 startPos, 
                 float3 lightDir, 
                 float maxDist,
                 texture3d<uint, access::read> indirection,
                 device uint* geoPool) // Linear Buffer Access
{
    // Precomputes
    float3 rayInvDir = 1.0f / (lightDir + 1e-5f);
    int3 step = int3(sign(lightDir));
    float3 deltaDist = abs(rayInvDir); // For brick-sized grid (unit 1.0 in brick space)

    // Convert to Brick Space (1 unit = 8 voxels)
    float3 rayStart = startPos / 8.0f;
    int3 mapPos = int3(floor(rayStart));
    
    // Setup Coarse DDA
    float3 sideDist = (sign(lightDir) * (float3(mapPos) - rayStart) + (sign(lightDir) * 0.5f) + 0.5f) * deltaDist;

    // Track distance to handle MaxDist and SDF skipping logic
    // We infer t_current from the DDA state when needed to avoid redundant registers
    float distTraveled = 0.0f; 

    // Max 256 steps through the coarse grid (covers ~2048 world units)
    for (int i = 0; i < 256; i++) {
        
        // Bounds Check
        if (mapPos.x < 0 || mapPos.y < 0 || mapPos.z < 0 ||
            mapPos.x >= indirection.get_width() || 
            mapPos.y >= indirection.get_height() || 
            mapPos.z >= indirection.get_depth()) return false;

        // Fetch Cell Data
        uint val = indirection.read(uint3(mapPos)).r;
        uint distByte = val >> 24;      
        uint cellData = val & 0x00FFFFFF; 

        // 1. SDF SKIP
        if (distByte > 1) {
            // We can skip (Dist - 1) bricks safely. 
            // In brick coordinates, this is simply the byte value.
            float skipVal = float(distByte) - 0.5f; 
            
            // Advance DDA state
            // Reconstruct exact position to prevent DDA drift over large distances
            distTraveled += skipVal; 
            
            float3 p = rayStart + lightDir * distTraveled;
            mapPos = int3(floor(p));
            sideDist = (sign(lightDir) * (float3(mapPos) - p) + (sign(lightDir) * 0.5f) + 0.5f) * deltaDist;
            continue; 
        }

        // 2. SOLID HIT
        if (cellData == FLAG_SOLID_GENERIC) return true;

        // 3. MIXED HIT (Brick)
        if (cellData >= IND_OFFSET) {
            uint brickIdx = cellData - IND_OFFSET;
            uint geoBase = brickIdx * 16; // 16 uints per brick

            // Calculate precise entry point into this brick
            // t_entry = (sideDist - deltaDist).max_component?
            // A more robust way given we might have drifted:
            // Center the local coordinate calculation on the current mapPos
            float3 brickWorldPos = float3(mapPos) * 8.0f;
            
            // We need the ray's current position *at entry* to the brick.
            // Reconstruct: startPos + dir * t
            // Approximating t from sideDist is fastest:
            float3 tStep = sideDist - deltaDist;
            float tEntry = max(max(tStep.x, tStep.y), tStep.z);
            if (tEntry < 0.0f) tEntry = 0.0f;

            float3 worldEntry = startPos + lightDir * (tEntry * 8.0f + 0.001f);
            float3 localP = worldEntry - brickWorldPos;
            
            // Clamp to 0..8 range to handle precision issues
            localP = clamp(localP, 0.0f, 7.99f);

            // Setup Sub-Chunk DDA (2x2x4 grid)
            // Sub-chunk size: (4, 4, 2)
            int3 subMapPos = int3(localP / float3(4.0f, 4.0f, 2.0f));
            subMapPos = clamp(subMapPos, int3(0), int3(1, 1, 3));

            float3 subSize = float3(4.0f, 4.0f, 2.0f);
            float3 subDelta = abs(rayInvDir * subSize); // Scaled delta
            float3 subRayStart = localP / subSize;
            
            float3 subSide = (sign(lightDir) * (float3(subMapPos) - subRayStart) + (sign(lightDir) * 0.5f) + 0.5f) * subDelta;

            // Traverse the 16 sub-chunks
            for(int k=0; k<8; k++) { // Max diagonal steps is 1+1+3 = 5, safety 8
                
                // Linear Index: x + y*2 + z*4
                uint subIdx = subMapPos.x + (subMapPos.y * 2) + (subMapPos.z * 4);
                
                // Read from Buffer
                uint packed = geoPool[geoBase + subIdx];

                if (packed != 0) {
                    // Localize position to this 4x4x2 block
                    float3 subOffset = float3(subMapPos.x*4, subMapPos.y*4, subMapPos.z*2);
                    float3 bitLocalP = localP - subOffset;
                    
                    // Optimized Register Check
                    if (checkBitLocal_Shadow(packed, bitLocalP, lightDir, rayInvDir, step)) {
                        return true;
                    }
                }

                // Step Sub-DDA
                if (subSide.x < subSide.y) {
                    if (subSide.x < subSide.z) {
                        subSide.x += subDelta.x; subMapPos.x += step.x;
                    } else {
                        subSide.z += subDelta.z; subMapPos.z += step.z;
                    }
                } else {
                    if (subSide.y < subSide.z) {
                        subSide.y += subDelta.y; subMapPos.y += step.y;
                    } else {
                        subSide.z += subDelta.z; subMapPos.z += step.z;
                    }
                }
                
                // Exit Brick Check
                if(subMapPos.x < 0 || subMapPos.x > 1 || 
                   subMapPos.y < 0 || subMapPos.y > 1 || 
                   subMapPos.z < 0 || subMapPos.z > 3) break;
            }
        }

        // 4. STEP COARSE DDA
        if (sideDist.x < sideDist.y) {
            if (sideDist.x < sideDist.z) {
                sideDist.x += deltaDist.x; mapPos.x += step.x;
            } else {
                sideDist.z += deltaDist.z; mapPos.z += step.z;
            }
        } else {
            if (sideDist.y < sideDist.z) {
                sideDist.y += deltaDist.y; mapPos.y += step.y;
            } else {
                sideDist.z += deltaDist.z; mapPos.z += step.z;
            }
        }
    }
    
    return false;
}