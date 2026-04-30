// Material Generation
// Procedural terrain generation and brick filling.

#include "ShaderTypes.slang"
#include "Settings.slang"
#include "TerrainGen.slang"

RWStructuredBuffer<uint64_t> resultBuffer;
StructuredBuffer<SectorWorkItem> workItems;
RWStructuredBuffer<uint64_t> resultBufferStreaming;
StructuredBuffer<BrickWorkItem> workList;
RWStructuredBuffer<SectorInfo> sectorBuffer;
RWStructuredBuffer<uint64_t> occupancyBuffer;
RWStructuredBuffer<uint8_t> dataBuffer;

uniform int3 worldOrigin;
uniform uint totalItems;

Texture3D<uint> indirection;
RWStructuredBuffer<uint64_t> geoPool;
RWStructuredBuffer<uint8_t> matPool;

[ForceInline]
inline uint GetLinearIndex(uint3 pos, uint sizeX, uint sizeY) {
    return pos.x + (pos.y * sizeX) + (pos.z * sizeX * sizeY);
}

[ForceInline]
inline int countbits_64(uint64_t v) {
    return countbits(v);
}

// Procedural material ID based on position

[ForceInline]
inline uint8_t get_procedural_material_id(float3 pos) {
    int y = (int)pos.y;
    
    // Bedrock at bottom
    if (y < 4) {
        if (y == 0) return MAT_BEDROCK;
        if (simplex2D(pos.x * 0.1f, pos.z * 0.1f) > 0.0f) return MAT_BEDROCK;
    }

    // Biome calculation
    float biomeNoise = simplex2D(pos.x * 0.003f, pos.z * 0.003f);
    
    float baseHeight = 140.0f;
    float mountain = (biomeNoise + 1.0f) * 0.5f * 200.0f;
    float approxSurfaceY = baseHeight + (simplex2D(pos.x * 0.005f, pos.z * 0.005f) * 20.0f);
    if (biomeNoise > 0.4f) approxSurfaceY += mountain;

    int depthFromSurface = (int)(approxSurfaceY - pos.y);

    // Surface layers
    if (depthFromSurface >= 0 && depthFromSurface < 5) {
        if (biomeNoise < -0.2f) {
            return uint8_t((depthFromSurface == 0) ? MAT_SAND : MAT_SANDSTONE);
        }
        else if (biomeNoise > 0.5f && y > 260) {
            return MAT_BRICK;
        }
        else {
            return uint8_t((depthFromSurface == 0) ? MAT_GRASS : MAT_DIRT);
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

[ForceInline]
inline uint GetSectorIndex(uint3 pos, uint sx, uint sy) {
    return pos.x + (pos.y * sx) + (pos.z * sx * sy);
}

[ForceInline]
inline uint64_t pack_4x4x4_block(float3 startPos) {
    uint64_t packed = 0;
    for(int z=0; z<4; z++) {
        for(int y=0; y<4; y++) {
            for(int x=0; x<4; x++) {
                if(Evaluate(startPos.x + x, startPos.y + y, startPos.z + z) > 0.0f) {
                    packed |= (1ULL << (x + y * 4 + z * 16));
                }
            }
        }
    }
    return packed;
}

// Kernel: Analyze sectors for initial generation
[shader("compute")]
[numthreads(4, 4, 4)]
void XMap_AnalyzeSectors(uint3 gid : SV_DispatchThreadID)
{
    uint sx = IND_X / 4;
    uint sy = IND_Y / 4;
    uint sz = IND_Z / 4;
    if (gid.x >= sx || gid.y >= sy || gid.z >= sz) return;

    uint sectorIndex = GetSectorIndex(gid, sx, sy);
    float3 sectorWorldPos = float3(gid) * 32.0f;
    uint64_t activeBricksMask = 0;

    for(int i = 0; i < 64; i++) {
        int bx = i & 3;
        int bz = (i >> 2) & 3;
        int by = (i >> 4) & 3;
        float3 brickPos = sectorWorldPos + float3(bx, by, bz) * 8.0f;

        bool active = false;
        for(int dz = 0; dz < 8; dz += 3) {
            for(int dy = 0; dy < 8; dy += 3) {
                for(int dx = 0; dx < 8; dx += 3) {
                    if (Evaluate(brickPos.x + dx, brickPos.y + dy, brickPos.z + dz) > 0.0f) {
                        active = true;
                        break;
                    }
                }
                if(active) break;
            }
            if(active) break;
        }
        
        if (active) activeBricksMask |= (1ULL << i);
    }
    resultBuffer[sectorIndex] = activeBricksMask;
}

// Kernel: Analyze sectors for streaming
[shader("compute")]
[numthreads(64, 1, 1)]
void XMap_AnalyzeStreaming(uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= totalItems) return;

    SectorWorkItem item = workItems[gid.x];
    float3 sectorWorldPos = float3(item.worldX, item.worldY, item.worldZ) * 32.0f;
    uint64_t activeBricksMask = 0;

    for(int i = 0; i < 64; i++) {
        int bx = i & 3;
        int bz = (i >> 2) & 3;
        int by = (i >> 4) & 3;
        float3 brickPos = sectorWorldPos + float3(bx, by, bz) * 8.0f;

        bool active = false;
        for(int dz = 0; dz < 8; dz += 3) {
            for(int dy = 0; dy < 8; dy += 3) {
                for(int dx = 0; dx < 8; dx += 3) {
                    if (Evaluate(brickPos.x + dx, brickPos.y + dy, brickPos.z + dz) > 0.0f) {
                        active = true;
                        break;
                    }
                }
                if(active) break;
            }
            if(active) break;
        }
        
        if (active) activeBricksMask |= (1ULL << i);
    }
    resultBufferStreaming[gid.x] = activeBricksMask;
}

// Kernel: Fill brick data
[shader("compute")]
[numthreads(64, 1, 1)]
void XMap_FillBricks(uint3 gid : SV_DispatchThreadID)
{
    uint gid_val = gid.x;
    uint threadgroupID = gid_val / 64;
    uint workItemIndex = threadgroupID / 8;
    uint subBrickIndex = threadgroupID % 8;
    uint threadID_local = gid_val % 64;
    
    BrickWorkItem item = workList[workItemIndex];
    
    uint sx = IND_X / 4;
    uint sy = IND_Y / 4;
    
    uint s_rem = item.sectorIndex;
    uint s_x = s_rem % sx;
    s_rem /= sx;
    uint s_y = s_rem % sy;
    uint s_z = s_rem / sy;
    
    float3 sectorPos = float3(int(s_x) + worldOrigin.x,
                               int(s_y) + worldOrigin.y,
                               int(s_z) + worldOrigin.z) * 32.0f;
    
    uint b_x = item.localBrickIndex & 3;
    uint b_z = (item.localBrickIndex >> 2) & 3;
    uint b_y = (item.localBrickIndex >> 4) & 3;
    
    float3 brickPos = sectorPos + float3(b_x, b_y, b_z) * 8.0f;
    
    uint sb_x = subBrickIndex & 1;
    uint sb_z = (subBrickIndex >> 1) & 1;
    uint sb_y = (subBrickIndex >> 2) & 1;
    
    float3 subBrickPos = brickPos + float3(sb_x, sb_y, sb_z) * 4.0f;
    
    uint v_x = threadID_local & 3;
    uint v_z = (threadID_local >> 2) & 3;
    uint v_y = (threadID_local >> 4) & 3;
    
    float3 voxelPos = subBrickPos + float3(v_x, v_y, v_z);
    
    float density = Evaluate(voxelPos.x, voxelPos.y, voxelPos.z);
    bool isSolid = density > 0.0f;
    
    static groupshared uint groupMaskHigh;
    static groupshared uint groupMaskLow;
    
    if (threadID_local == 0) {
        groupMaskHigh = 0;
        groupMaskLow = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (isSolid) {
        if (threadID_local < 32) {
            InterlockedOr(groupMaskLow, (1u << threadID_local));
        } else {
            InterlockedOr(groupMaskHigh, (1u << (threadID_local - 32)));
        }
        
        uint8_t matID = get_procedural_material_id(voxelPos);
        uint64_t finalDataIdx = item.dataOffset + (subBrickIndex * 64) + threadID_local;
        dataBuffer[finalDataIdx] = matID;
    } else {
        uint64_t finalDataIdx = item.dataOffset + (subBrickIndex * 64) + threadID_local;
        dataBuffer[finalDataIdx] = 0;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (threadID_local == 0) {
        uint64_t fullMask = (uint64_t(groupMaskHigh) << 32) | uint64_t(groupMaskLow);
        occupancyBuffer[item.occupancyOffset + subBrickIndex] = fullMask;
    }
}

// Kernel: Fill dynamic atlases
[shader("compute")]
[numthreads(8, 8, 8)]
void FillDynamicAtlases(uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= IND_X || gid.y >= IND_Y || gid.z >= IND_Z) return;

    uint index = indirection[gid];

    if (index < IND_OFFSET) return;

    uint brickIdx = index - IND_OFFSET;

    uint64_t geoBaseIdx = (uint64_t)brickIdx * 8;
    uint64_t matBaseIdx = (uint64_t)brickIdx * 512;

    float3 worldBase = float3(gid) * 8.0f;

    for(int z=0; z<2; z++) {
        for(int y=0; y<2; y++) {
            for(int x=0; x<2; x++) {
                float3 chunkPos = worldBase + float3(x*4, y*4, z*4);
                uint64_t packed = pack_4x4x4_block(chunkPos);
                uint localChunkIdx = x + (y * 2) + (z * 4);
                geoPool[geoBaseIdx + localChunkIdx] = packed;
            }
        }
    }

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
