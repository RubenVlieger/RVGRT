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
    texture3d<float, access::write> GIdata [[texture(0)]],
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

    GIdata.write(float4((float3)accumulatedColor.xyz, 1.0f), gid);
}

#define GI_PACKING_SCALE 8.0h

kernel void CoarseArray_GlobalIlluminate(
    texture3d<float, access::sample> GIdata_Read [[texture(0)]], // Bound as SAMPLE
    texture3d<uint, access::read> packedTex [[texture(1)]],
    texture3d<float, access::sample> csdf [[texture(2)]], 
    texture2d<float, access::sample> texturepack [[texture(3)]],
    constant float3& c_sunDir2 [[buffer(4)]],
    constant uint& frameNumber [[buffer(5)]],
    constant uint64_t& offset [[buffer(6)]],
    texture3d<float, access::write> GIdata_Write [[texture(7)]], // Bound as WRITE
    uint idx_local [[thread_position_in_grid]])
{
    uint64_t idx = idx_local + offset;
    if (idx >= GI_SIZE) return;

    // ... Coordinate setup ...
    uint z = idx / (GI_SIZEX * GI_SIZEY);
    uint temp = idx % (GI_SIZEX * GI_SIZEY);
    uint y = temp / GI_SIZEX;
    uint x = temp % GI_SIZEX;
    uint3 gid = uint3(x, y, z);

    // If solid, write 0 (encoded 0 is just 0)
    if (isCoarseBlockSolid(x, y, z, packedTex)) { 
        GIdata_Write.write(float4(0.0f), gid);
        return; 
    }

    uint random_state = init_random_state(idx, frameNumber);
    float3 worldPos = make_float3((x + 0.5f) * COARSENESSGI, 
                                  (y + 0.5f) * COARSENESSGI, 
                                  (z + 0.5f) * COARSENESSGI);

    half3 newSample = make_half3(0.0h);

    // --- 1. Calculate Lighting (HDR Space) ---
    // Direct Light
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.001f, packedTex, csdf);
    if (!shadowHit.hit) {
        newSample += c_sunColor / 8.0f; 
    }

    // Indirect Bounce
    float3 randomDir = random_direction_in_sphere(random_state);
    hitInfo bounceHit = trace(worldPos, randomDir, 0.001f, packedTex, csdf);

    if (bounceHit.hit) {
        constexpr sampler s(address::clamp_to_edge, filter::linear);
        float3 uvw = bounceHit.pos / float3(SIZEX, SIZEY, SIZEZ);
        
        // SAMPLE NEIGHBOR (Returns 0..1)
        half4 rawNeighbor = (half4)GIdata_Read.sample(s, uvw);
        
        // DECODE: Convert 0..1 back to 0..8 HDR
        half3 neighborLight = rawNeighbor.rgb * GI_PACKING_SCALE;

        half3 surfaceAlbedo = sampleTexture(bounceHit.uv, bounceHit.pos, texturepack);
        newSample += (neighborLight * surfaceAlbedo);
    } else {
        newSample += sampleSky(randomDir, c_sunDir2);
    }

    // --- 2. Temporal Accumulation ---
    const half LEARNING_RATE = 0.04h;

    // READ HISTORY (Returns 0..1)
    half4 prevRaw = (half4)GIdata_Read.read(gid);
    
    // DECODE HISTORY
    half3 prevHDR = prevRaw.rgb * GI_PACKING_SCALE;
    
    // Blend in HDR space
    half3 finalHDR = lerp(prevHDR, newSample, LEARNING_RATE);
    
    // --- 3. Encode & Write ---
    // ENCODE: Divide by 8.0 to fit into unorm texture
    half3 finalEncoded = finalHDR / GI_PACKING_SCALE;
    
    // Clamp to 1.0 so we don't wrap around or look weird, though write() handles saturation usually
    finalEncoded = min(finalEncoded, half3(1.0h));

    GIdata_Write.write(float4((float3)finalEncoded, 1.0f), gid);
}