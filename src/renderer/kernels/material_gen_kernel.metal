#include <metal_stdlib>
#include "cumath.h"
#include "TerrainGeneration.h" 
#include "renderer/ShaderTypes.h"

using namespace metal;


uint8_t get_procedural_material_id(float3 pos) {
    int y = (int)pos.y;
    
    // 1. BEDROCK (Bottom of world)
    if (y < 4) {
        if (y == 0) return MAT_BEDROCK;
        if (simplex2D(pos.x * 0.1f, pos.z * 0.1f) > 0.0f) return MAT_BEDROCK;
    }

    // 2. BIOME CALCULATION
    float biomeNoise = simplex2D(pos.x * 0.003f, pos.z * 0.003f); 
    
    float baseHeight = 140.0f; 
    float mountain = (biomeNoise + 1.0f) * 0.5f * 200.0f;
    float approxSurfaceY = baseHeight + (simplex2D(pos.x * 0.005f, pos.z * 0.005f) * 20.0f);
    if (biomeNoise > 0.4f) approxSurfaceY += mountain;

    int depthFromSurface = (int)(approxSurfaceY - pos.y);

    // 3. SURFACE LAYERS
    if (depthFromSurface >= 0 && depthFromSurface < 5) {
        if (biomeNoise < -0.2f) {
            return (depthFromSurface == 0) ? MAT_SAND : MAT_SANDSTONE;
        }
        else if (biomeNoise > 0.5f && y > 260) {
            return MAT_BRICK;
        }
        else {
            return (depthFromSurface == 0) ? MAT_GRASS : MAT_DIRT;
        }
    }
    
    float oreNoise = simplex3D(pos.x * 0.12f, pos.y * 0.12f, pos.z * 0.12f);
    
    if (y < 16) {
        float diamNoise = simplex3D(pos.x * 0.15f - 50.0f, pos.y * 0.15f, pos.z * 0.15f);
        if (diamNoise > 0.82f) return MAT_DIAM_ORE;
    }
    if (y < 40) {
        float goldNoise = simplex3D(pos.x * 0.12f + 123.0f, pos.y * 0.12f, pos.z * 0.12f);
        if (goldNoise > 0.78f) return MAT_GOLD_ORE;
    }
    if (y < 120 && oreNoise > 0.72f) return MAT_IRON_ORE;
    if (oreNoise > 0.65f) return MAT_COAL_ORE;

    return MAT_STONE;
}


// -----------------------------------------------------------------------------
// KERNEL 1: ANALYSIS (Sparse Structure Detection)
// -----------------------------------------------------------------------------
kernel void AnalyzeWorldStructure(
    device uint* statusGrid [[buffer(0)]], 
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= IND_X || gid.y >= IND_Y || gid.z >= IND_Z) return;

    uint idx = (gid.z * IND_Y + gid.y) * IND_X + gid.x;
    float3 basePos = float3(gid) * 8.0f;

    bool seenSolid = false;
    bool seenAir = false;

    // 1. Check Corners
    float v0 = Evaluate(basePos.x, basePos.y, basePos.z);
    if(v0 > 0.0f) seenSolid = true; else seenAir = true;

    float v1 = Evaluate(basePos.x + 7, basePos.y + 7, basePos.z + 7);
    if(v1 > 0.0f) seenSolid = true; else seenAir = true;

    if(seenSolid && seenAir) { statusGrid[idx] = 2; return; }

    // 2. Linear scan
    for(int z = 0; z < 8; z++) {
        for(int y = 0; y < 8; y++) {
            for(int x = 0; x < 8; x++) {
                if ((x==0 && y==0 && z==0) || (x==7 && y==7 && z==7)) continue;

                float val = Evaluate(basePos.x + x, basePos.y + y, basePos.z + z);
                
                if (val > 0.0f) seenSolid = true; 
                else seenAir = true;

                if (seenSolid && seenAir) {
                    statusGrid[idx] = 2; // Mixed
                    return; 
                }
            }
        }
    }
    statusGrid[idx] = seenSolid ? 1 : 0;
}


// -----------------------------------------------------------------------------
// HELPER: Pack 4x4x4 voxels into one uint64 (ulong)
// -----------------------------------------------------------------------------
ulong pack_4x4x4_block(float3 startPos) {
    ulong packed = 0;
    
    // Linearize: x + y*4 + z*16
    for(int z=0; z<4; z++) {
        for(int y=0; y<4; y++) {
            for(int x=0; x<4; x++) {
                if(Evaluate(startPos.x + x, startPos.y + y, startPos.z + z) > 0.0f) {
                    // Shift using 64-bit literal to ensure correct width
                    packed |= (1UL << (x + y * 4 + z * 16));
                }
            }
        }
    }
    return packed;
}

// -----------------------------------------------------------------------------
// KERNEL 2: FILL (Population of Atlases)
// -----------------------------------------------------------------------------
kernel void FillDynamicAtlases(
    texture3d<uint, access::read>  indirection [[texture(0)]],
    device uint* geoPool                       [[buffer(0)]], 
    device uchar* matPool                      [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= IND_X || gid.y >= IND_Y || gid.z >= IND_Z) return;

    uint index = indirection.read(gid).r;

    if (index < IND_OFFSET) return;

    uint brickIdx = index - IND_OFFSET;

    // Geometry: 8 ulongs per brick (2x2x2 chunks of 4x4x4)
    // In the uint buffer, this is 16 uints.
    // We cast to ulong* for easier writing.
    device ulong* geoPool64 = (device ulong*)geoPool;
    uint geoBaseIdx = brickIdx * 8; 

    // Material: 512 bytes per brick
    uint matBaseIdx = brickIdx * 512;

    float3 worldBase = float3(gid) * 8.0f;

    // 1. Fill Geometry Pool (Loop through the 8 sub-chunks)
    // Layout: 2x2x2
    for(int z=0; z<2; z++) {
        for(int y=0; y<2; y++) {
            for(int x=0; x<2; x++) {
                float3 chunkPos = worldBase + float3(x*4, y*4, z*4);
                
                // Pack 64 bits (4x4x4)
                ulong packed = pack_4x4x4_block(chunkPos);
                
                uint localChunkIdx = x + (y * 2) + (z * 4);
                
                geoPool64[geoBaseIdx + localChunkIdx] = packed;
            }
        }
    }

    // 2. Fill Material Pool (Loop through 512 voxels)
    for(int z=0; z<8; z++) {
        for(int y=0; y<8; y++) {
            for(int x=0; x<8; x++) {
                float3 voxelPos = worldBase + float3(x,y,z);
                uint8_t matID = 0;
                
                if (Evaluate(voxelPos.x, voxelPos.y, voxelPos.z) > 0.0f) {
                    matID = get_procedural_material_id(voxelPos);
                }
                
                uint localVoxelIdx = x + (y * 8) + (z * 64);
                
                matPool[matBaseIdx + localVoxelIdx] = matID;
            }
        }
    }
}