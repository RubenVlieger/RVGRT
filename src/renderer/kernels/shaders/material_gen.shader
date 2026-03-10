#include <metal_stdlib>
#include "cumath.h"
#include "shader_macros.h"
#include "TerrainGeneration.h"
#include "renderer/ShaderTypes.h"

#if defined(PLATFORM_METAL)
using namespace metal;
#endif

// ============================================================================
// KERNELS: Material Generation
// 
// Procedural terrain generation and brick filling.
// Generates voxel data on-demand for the streaming world.
// ============================================================================

inline uint GetLinearIndex(uint3 pos, uint sizeX, uint sizeY) {
    return pos.x + (pos.y * sizeX) + (pos.z * sizeX * sizeY);
}

#if defined(PLATFORM_METAL)
inline int countbits_64(ulong v) {
    return popcount(v);
}
#else
__device__ inline int countbits_64(ulong v) {
    return __popcll(v);
}
#endif

// Procedural material ID based on position
uint8_t get_procedural_material_id(float3 pos) {
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

inline uint GetSectorIndex(uint3 pos, uint sx, uint sy) {
    return pos.x + (pos.y * sx) + (pos.z * sx * sy);
}

// Kernel: Analyze sectors for initial generation
KERNEL(XMap_AnalyzeSectors)(
    PARAM_BUFFER(device uint64_t, resultBuffer, 0),
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    uint3 gid = GET_GID();
    uint sx = IND_X / 4;
    uint sy = IND_Y / 4;
    uint sz = IND_Z / 4;
    if (gid.x >= sx || gid.y >= sy || gid.z >= sz) return;
#else
    int3 gid = GET_GID();
    uint sx = IND_X / 4;
    uint sy = IND_Y / 4;
    uint sz = IND_Z / 4;
    if (gid.x >= sx || gid.y >= sy || gid.z >= sz) return;
#endif

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
        
        if (active) activeBricksMask |= (1UL << i);
    }
    resultBuffer[sectorIndex] = activeBricksMask;
}

// Kernel: Analyze sectors for streaming
KERNEL(XMap_AnalyzeStreaming)(
    PARAM_BUFFER(device SectorWorkItem, workItems, 0),
    PARAM_BUFFER(device uint64_t, resultBuffer, 1),
    PARAM_CONSTANT(uint, totalItems, 2),
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    uint gid = GET_GID().x;
#else
    int gid = GET_GID().x;
#endif
    if (gid >= totalItems) return;

    SectorWorkItem item = workItems[gid];
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
        
        if (active) activeBricksMask |= (1UL << i);
    }
    resultBuffer[gid] = activeBricksMask;
}

// Kernel: Fill brick data
#if defined(PLATFORM_METAL)
KERNEL(XMap_FillBricks)(
    PARAM_BUFFER(device BrickWorkItem, workList, 0),
    PARAM_BUFFER(device SectorInfo, sectorBuffer, 1),
    PARAM_BUFFER(device uint64_t, occupancyBuffer, 2),
    PARAM_BUFFER(device uchar, dataBuffer, 3),
    PARAM_CONSTANT(int3, worldOrigin, 4),
    
    uint groupID [[threadgroup_position_in_grid]],
    uint threadID [[thread_position_in_threadgroup]]
)
#else
KERNEL(XMap_FillBricks)(
    PARAM_BUFFER(device BrickWorkItem, workList, 0),
    PARAM_BUFFER(device SectorInfo, sectorBuffer, 1),
    PARAM_BUFFER(device uint64_t, occupancyBuffer, 2),
    PARAM_BUFFER(device uchar, dataBuffer, 3),
    PARAM_CONSTANT(int3, worldOrigin, 4)
)
#endif
{
#if defined(PLATFORM_METAL)
    uint workItemIndex = groupID / 8;
    uint subBrickIndex = groupID % 8;
    uint threadID_local = threadID;
#else
    uint workItemIndex = blockIdx.x / 8;
    uint subBrickIndex = blockIdx.x % 8;
    uint threadID_local = threadIdx.x;
#endif
    
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
    
#if defined(PLATFORM_METAL)
    threadgroup atomic_uint groupMaskHigh;
    threadgroup atomic_uint groupMaskLow;
    
    if (threadID_local == 0) {
        atomic_store_explicit(&groupMaskHigh, 0, memory_order_relaxed);
        atomic_store_explicit(&groupMaskLow, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (isSolid) {
        if (threadID_local < 32) {
            atomic_fetch_or_explicit(&groupMaskLow, (1u << threadID_local), memory_order_relaxed);
        } else {
            atomic_fetch_or_explicit(&groupMaskHigh, (1u << (threadID_local - 32)), memory_order_relaxed);
        }
        
        uint8_t matID = get_procedural_material_id(voxelPos);
        uint64_t finalDataIdx = item.dataOffset + (subBrickIndex * 64) + threadID_local;
        dataBuffer[finalDataIdx] = matID;
    } else {
        uint64_t finalDataIdx = item.dataOffset + (subBrickIndex * 64) + threadID_local;
        dataBuffer[finalDataIdx] = 0;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (threadID_local == 0) {
        uint low = atomic_load_explicit(&groupMaskLow, memory_order_relaxed);
        uint high = atomic_load_explicit(&groupMaskHigh, memory_order_relaxed);
        uint64_t fullMask = (uint64_t(high) << 32) | uint64_t(low);
        
        occupancyBuffer[item.occupancyOffset + subBrickIndex] = fullMask;
    }
#else
    // CUDA path
    __shared__ uint groupMaskHigh;
    __shared__ uint groupMaskLow;
    
    if (threadID_local == 0) {
        groupMaskHigh = 0;
        groupMaskLow = 0;
    }
    __syncthreads();
    
    if (isSolid) {
        if (threadID_local < 32) {
            atomicOr(&groupMaskLow, (1u << threadID_local));
        } else {
            atomicOr(&groupMaskHigh, (1u << (threadID_local - 32)));
        }
        
        uint8_t matID = get_procedural_material_id(voxelPos);
        uint64_t finalDataIdx = item.dataOffset + (subBrickIndex * 64) + threadID_local;
        dataBuffer[finalDataIdx] = matID;
    } else {
        uint64_t finalDataIdx = item.dataOffset + (subBrickIndex * 64) + threadID_local;
        dataBuffer[finalDataIdx] = 0;
    }
    
    __syncthreads();
    
    if (threadID_local == 0) {
        uint64_t fullMask = (uint64_t(groupMaskHigh) << 32) | uint64_t(groupMaskLow);
        occupancyBuffer[item.occupancyOffset + subBrickIndex] = fullMask;
    }
#endif
}

ulong pack_4x4x4_block(float3 startPos) {
    ulong packed = 0;
    for(int z=0; z<4; z++) {
        for(int y=0; y<4; y++) {
            for(int x=0; x<4; x++) {
                if(Evaluate(startPos.x + x, startPos.y + y, startPos.z + z) > 0.0f) {
                    packed |= (1UL << (x + y * 4 + z * 16));
                }
            }
        }
    }
    return packed;
}

// Kernel: Fill dynamic atlases
KERNEL(FillDynamicAtlases)(
    PARAM_TEXTURE_READ(texture3d<uint, access::read>, indirection, 0),
    PARAM_BUFFER(device uint, geoPool, 0),
    PARAM_BUFFER(device uchar, matPool, 1),
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    uint3 gid = GET_GID();
#else
    int3 gid = GET_GID();
#endif
    if (gid.x >= IND_X || gid.y >= IND_Y || gid.z >= IND_Z) return;

    uint index = TEX_READ_3D(indirection, gid);

    if (index < IND_OFFSET) return;

    uint brickIdx = index - IND_OFFSET;

#if defined(PLATFORM_METAL)
    device ulong* geoPool64 = (device ulong*)geoPool;
#else
    ulong* geoPool64 = (ulong*)geoPool;
#endif
    uint64_t geoBaseIdx = (uint64_t)brickIdx * 8;
    uint64_t matBaseIdx = (uint64_t)brickIdx * 512;

    float3 worldBase = float3(gid) * 8.0f;

    for(int z=0; z<2; z++) {
        for(int y=0; y<2; y++) {
            for(int x=0; x<2; x++) {
                float3 chunkPos = worldBase + float3(x*4, y*4, z*4);
                ulong packed = pack_4x4x4_block(chunkPos);
                uint localChunkIdx = x + (y * 2) + (z * 4);
                geoPool64[geoBaseIdx + localChunkIdx] = packed;
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
