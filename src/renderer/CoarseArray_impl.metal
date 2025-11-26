#include <metal_stdlib>
#include "cumath.h"
#include "CoarseArray.h"
#include "raytracing_functions.h"
#include "TerrainGeneration.h"
using namespace metal;

inline uint3 indexTo3D(uint index, uint width, uint height) {
    uint z = index / (width * height);
    uint temp = index % (width * height);
    uint y = temp / width;
    uint x = temp % width;
    return uint3(x, y, z);
}

kernel void CoarseArray_computeDistX(texture3d<uint, access::read> packedTex [[texture(0)]], 
                                     texture3d<float, access::write> outDistX [[texture(1)]],
                                     uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= SDF_SIZEX || gid.y >= SDF_SIZEY || gid.z >= SDF_SIZEZ) return;

    if (isCoarseBlockSolid(gid.x, gid.y, gid.z, packedTex)) {
        outDistX.write(float4(0.0f), gid);
        return;
    }
    
    int min_d = SDF_MAX_DIST;
    
    for (int i = 1; i <= SDF_MAX_DIST; ++i) {
        bool left  = ((int)gid.x - i >= 0)           &&   isCoarseBlockSolid(gid.x - i, gid.y, gid.z, packedTex);
        bool right = ((int)gid.x + i < (int)SDF_SIZEX) && isCoarseBlockSolid(gid.x + i, gid.y, gid.z, packedTex);
        
        if (left || right) {
            min_d = i;
            break;
        }
    }
    outDistX.write(float4((float)min_d), gid);
}



kernel void CoarseArray_computeDistY(texture3d<float, access::sample> inDistX [[texture(0)]],
                                     texture3d<float, access::write> outDistY [[texture(1)]],
                                     uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= SDF_SIZEX || gid.y >= SDF_SIZEY || gid.z >= SDF_SIZEZ) return;

    float current_dx = inDistX.read(gid).r;
    
    if (current_dx == 0) {
        outDistY.write(float4(0.0f), gid);
        return;
    }

    float min_dist_sq = current_dx * current_dx;

    for (int y_offset = 1; y_offset <= SDF_MAX_DIST; ++y_offset) {
        if ((float)(y_offset * y_offset) >= min_dist_sq) break;
        
        if ((int)gid.y >= y_offset) {
            float d = inDistX.read(uint3(gid.x, gid.y - y_offset, gid.z)).r;
            min_dist_sq = fmin(min_dist_sq, d*d + (float)(y_offset * y_offset));
        }
        if ((int)gid.y + y_offset < (int)SDF_SIZEY) {
            float d = inDistX.read(uint3(gid.x, gid.y + y_offset, gid.z)).r;
            min_dist_sq = fmin(min_dist_sq, d*d + (float)(y_offset * y_offset));
        }
    }
    float result = fmin(sqrt(min_dist_sq), (float)SDF_MAX_DIST);
    outDistY.write(float4(result), gid);
}


kernel void CoarseArray_computeDistZ(texture3d<float, access::sample> inDistXY [[texture(0)]],
                                     texture3d<float, access::write> outDistZ [[texture(1)]],
                                     uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= SDF_SIZEX || gid.y >= SDF_SIZEY || gid.z >= SDF_SIZEZ) return;

    float current_dxy = inDistXY.read(gid).r;
    if (current_dxy == 0) {
        outDistZ.write(float4(0.0f), gid);
        return;
    }

    float min_dist_sq = (float)current_dxy * (float)current_dxy;
    
    for (int z_offset = 1; z_offset <= SDF_MAX_DIST; ++z_offset) {
        if ((float)(z_offset * z_offset) >= min_dist_sq) break;

        if ((int)gid.z >= z_offset) {
            float d = inDistXY.read(uint3(gid.x, gid.y, gid.z - z_offset)).r;
            min_dist_sq = fmin(min_dist_sq, d*d + (float)(z_offset * z_offset));
        }
        if ((int)gid.z + z_offset < (int)SDF_SIZEZ) {
            float d = inDistXY.read(uint3(gid.x, gid.y, gid.z + z_offset)).r;
            min_dist_sq = fmin(min_dist_sq, d*d + (float)(z_offset * z_offset));
        }
    }
    float result = fmin(sqrt(min_dist_sq), (float)SDF_MAX_DIST);
    outDistZ.write(float4(result), gid);
}