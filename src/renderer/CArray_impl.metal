#include <metal_stdlib>
#include "cumath.h"
#include "TerrainGeneration.h" 

using namespace metal;


bool is_solid_at(float x, float y, float z) {
    float density = Evaluate(x, y, z);
    return density > 0.7f;
}


kernel void GeneratePackedWorld(
    texture3d<uint, access::write> voxelTex [[texture(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= voxelTex.get_width() || 
        gid.y >= voxelTex.get_height() || 
        gid.z >= voxelTex.get_depth()) return;


    float3 origin = float3(gid) * 2.0f;

    uint packedByte = 0;

    if (is_solid_at(origin.x + 0, origin.y + 0, origin.z + 0)) packedByte |= (1 << 0);
    if (is_solid_at(origin.x + 1, origin.y + 0, origin.z + 0)) packedByte |= (1 << 1);
    if (is_solid_at(origin.x + 0, origin.y + 1, origin.z + 0)) packedByte |= (1 << 2);
    if (is_solid_at(origin.x + 1, origin.y + 1, origin.z + 0)) packedByte |= (1 << 3);

    if (is_solid_at(origin.x + 0, origin.y + 0, origin.z + 1)) packedByte |= (1 << 4);
    if (is_solid_at(origin.x + 1, origin.y + 0, origin.z + 1)) packedByte |= (1 << 5);
    if (is_solid_at(origin.x + 0, origin.y + 1, origin.z + 1)) packedByte |= (1 << 6);
    if (is_solid_at(origin.x + 1, origin.y + 1, origin.z + 1)) packedByte |= (1 << 7);

    voxelTex.write(uint4(packedByte, 0, 0, 0), gid);
}