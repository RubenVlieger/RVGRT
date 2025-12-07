#include <metal_stdlib>
#include "cumath.h"
#include "TerrainGeneration.h" 

using namespace metal;

bool is_solid_at(float x, float y, float z) {
    float density = Evaluate(x, y, z);
    return density > 0.7f;
}


// Packs a 4x4x2 volume into a single uint32
// Layout: X runs fastest (0-3), then Y (0-3), then Z (0-1)
kernel void GeneratePackedWorld(
    texture3d<uint, access::write> voxelTex [[texture(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= voxelTex.get_width() || 
        gid.y >= voxelTex.get_height() || 
        gid.z >= voxelTex.get_depth()) return;


    float3 basePos = float3(gid.x * 4, gid.y * 4, gid.z * 2);

    uint packedBlock = 0;
    
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            if (is_solid_at(basePos.x + x, basePos.y + y, basePos.z)) {
                packedBlock |= (1u << (x + y * 4));
            }
        }
    }

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            if (is_solid_at(basePos.x + x, basePos.y + y, basePos.z + 1)) {
                packedBlock |= (1u << (x + y * 4 + 16));
            }
        }
    }

    voxelTex.write(uint4(packedBlock, 0, 0, 0), gid);
}