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



kernel void CoarseArray_InitialGlobalIlluminate(
    texture3d<uint, access::write> GIdata [[texture(0)]],
    texture3d<uint, access::read> packedTex [[texture(1)]],
    texture3d<float, access::sample> csdf [[texture(2)]], 
    constant float3& c_sunDir2 [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= GI_SIZEX || gid.y >= GI_SIZEY || gid.z >= GI_SIZEZ) return;
    
    float3 worldPos = make_float3((gid.x + 0.5f) * COARSENESSGI, 
                                  (gid.y + 0.5f) * COARSENESSGI, 
                                  (gid.z + 0.5f) * COARSENESSGI);
    
    half3 accumulatedColor = make_half3(0.h);

    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.0001f, packedTex, csdf);
    if (!shadowHit.hit) {
        accumulatedColor = c_sunColor;
    }
    
    uint packedColor = ((uint)(accumulatedColor.x * 255.h) << 0) | 
                       ((uint)(accumulatedColor.y * 255.h) << 8) | 
                       ((uint)(accumulatedColor.z * 255.h) << 16) | 
                       (255u << 24);
                       
    GIdata.write(uint4(packedColor, 0, 0, 0), gid);
}

kernel void CoarseArray_GlobalIlluminate(
    texture3d<uint, access::read_write> GIdata_curr [[texture(0)]],
    texture3d<uint, access::read> packedTex [[texture(1)]],
    texture3d<float, access::sample> csdf [[texture(2)]], 
    texture2d<float, access::sample> texturepack [[texture(3)]],
    constant float3& c_sunDir2 [[buffer(4)]],
    constant uint& frameNumber [[buffer(5)]],
    constant uint64_t& offset [[buffer(6)]],
    uint idx_local [[thread_position_in_grid]])
{
    uint64_t idx = idx_local + offset;
    if (idx >= GI_SIZE) return;

    uint z = idx / (GI_SIZEX * GI_SIZEY);
    uint temp = idx % (GI_SIZEX * GI_SIZEY);
    uint y = temp / GI_SIZEX;
    uint x = temp % GI_SIZEX;
    uint3 gid = uint3(x, y, z);

    float3 worldPos = make_float3((x + 0.5f) * COARSENESSGI, 
                                  (y + 0.5f) * COARSENESSGI, 
                                  (z + 0.5f) * COARSENESSGI);

    if (isCoarseBlockSolid(x, y, z, packedTex)) { 
        GIdata_curr.write(uint4(0), gid);
        return; 
    }

    uint random_state = init_random_state(idx, frameNumber);
    half3 newSample = make_half3(0.0h);

    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.001f, packedTex, csdf);
    if (!shadowHit.hit) {
        newSample += c_sunColor;
    }

    float3 randomDir = random_direction_in_sphere(random_state);
    hitInfo bounceHit = trace(worldPos, randomDir, 0.001f, packedTex, csdf);

    if (bounceHit.hit) {
        int3 g = int3(floor(bounceHit.pos / (float)COARSENESSGI));
        if (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < (int)GI_SIZEX && g.y < (int)GI_SIZEY && g.z < (int)GI_SIZEZ) {
            
            uint prevSample = GIdata_curr.read(uint3(g)).r;
            
            half3 bouncedColor = make_half3((half)(prevSample & 255) / 255.0h, 
                                            (half)((prevSample >> 8) & 255) / 255.0h, 
                                            (half)((prevSample >> 16) & 255) / 255.0h);
            
            half3 surfaceAlbedo = sampleTexture(bounceHit.uv, bounceHit.pos, texturepack);
            newSample += (bouncedColor * surfaceAlbedo);
        }
    } else {
        newSample += sampleSky(randomDir, c_sunDir2);
    }

    const half LEARNING_RATE = 0.04h;
    uint prevData = GIdata_curr.read(gid).r;
    half3 previousColor = make_half3((half)(prevData & 255) / 255.h, 
                                     (half)((prevData >> 8) & 255) / 255.h, 
                                     (half)((prevData >> 16) & 255) / 255.h);
    
    half3 finalColor = lerp(previousColor, newSample, LEARNING_RATE);
    finalColor = fmin(finalColor, make_half3(2.0h));

    uint packedFinal = ((uint)(fmin(finalColor.x, 1.0h) * 255.h) << 0) | 
                       ((uint)(fmin(finalColor.y, 1.0h) * 255.h) << 8) | 
                       ((uint)(fmin(finalColor.z, 1.0h) * 255.h) << 16) | 
                       (255u << 24);

    GIdata_curr.write(uint4(packedFinal, 0, 0, 0), gid);
}