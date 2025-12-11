#include <metal_stdlib>
#include "cumath.h" 
using namespace metal;

// Helper to pack/unpack coordinates (0..255) into a single uint
inline uint packCoord(uint3 c) {
    return (c.x & 0xFF) | ((c.y & 0xFF) << 8) | ((c.z & 0xFF) << 16);
}

inline uint3 unpackCoord(uint p) {
    return uint3(p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF);
}

// 1. INIT: Identify "Seeds" (Solid/Mixed bricks)
kernel void JFA_Init(
    texture3d<uint, access::read> indirection [[texture(0)]],
    texture3d<uint, access::write> seedMap    [[texture(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= indirection.get_width() || gid.y >= indirection.get_height() || gid.z >= indirection.get_depth()) return;

    // Any non-zero value means this brick is interesting (Solid or Mixed)
    if (indirection.read(gid).r != 0) {
        seedMap.write(uint4(packCoord(gid), 0,0,0), gid);
    } else {
        // Mark as "Infinite" / Invalid
        seedMap.write(uint4(0xFFFFFFFF, 0,0,0), gid);
    }
}

// 2. UPDATE: Jump Flood Step
kernel void JFA_Step(
    texture3d<uint, access::read>  inputMap  [[texture(0)]],
    texture3d<uint, access::write> outputMap [[texture(1)]],
    constant int& stepWidth                  [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= inputMap.get_width() || gid.y >= inputMap.get_height() || gid.z >= inputMap.get_depth()) return;

    uint bestPacked = inputMap.read(gid).r;
    float bestDistSq = 999999.0f;
    
    // If current pixel has a seed, calc initial distance
    if (bestPacked != 0xFFFFFFFF) {
        float3 seedPos = float3(unpackCoord(bestPacked));
        float3 myPos = float3(gid);
        bestDistSq = dot(seedPos - myPos, seedPos - myPos);
    }

    // Check 3x3x3 neighbors at +/- stepWidth
    for(int z = -1; z <= 1; z++) {
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                if(x==0 && y==0 && z==0) continue;

                int3 samplePos = int3(gid) + int3(x, y, z) * stepWidth;
                
                // Bounds Check
                if (samplePos.x >= 0 && samplePos.x < int(inputMap.get_width()) &&
                    samplePos.y >= 0 && samplePos.y < int(inputMap.get_height()) &&
                    samplePos.z >= 0 && samplePos.z < int(inputMap.get_depth())) 
                {
                    uint neighborPacked = inputMap.read(uint3(samplePos)).r;
                    
                    if (neighborPacked != 0xFFFFFFFF) {
                        float3 seedPos = float3(unpackCoord(neighborPacked));
                        float3 myPos = float3(gid);
                        float distSq = dot(seedPos - myPos, seedPos - myPos);
                        
                        if (distSq < bestDistSq) {
                            bestDistSq = distSq;
                            bestPacked = neighborPacked;
                        }
                    }
                }
            }
        }
    }
    
    outputMap.write(uint4(bestPacked, 0,0,0), gid);
}

// 3. COMMIT: Write Distance into Top 8 bits of Indirection Grid
kernel void JFA_Commit(
    texture3d<uint, access::read>  finalSeedMap [[texture(0)]],
    texture3d<uint, access::read_write> indirection [[texture(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= indirection.get_width() || gid.y >= indirection.get_height() || gid.z >= indirection.get_depth()) return;

    uint packedSeed = finalSeedMap.read(gid).r;
    uint currentVal = indirection.read(gid).r;

    // If I am solid/mixed, distance is 0. (Top 8 bits remain 0)
    if (currentVal != 0) return;

    // I am empty. Calculate distance to nearest solid.
    if (packedSeed != 0xFFFFFFFF) {
        float3 seedPos = float3(unpackCoord(packedSeed));
        float3 myPos = float3(gid);

        int3 diff = abs(int3(seedPos) - int3(myPos));
        int chebDist = max(max(diff.x, diff.y), diff.z);
        
        uint distByte = min((uint)chebDist, 255u);
        
        // Pack into top 8 bits
        // Preserve lower 24 bits (though they should be 0 for empty cells)
        indirection.write(uint4(currentVal | (distByte << 24), 0,0,0), gid);
    }
}