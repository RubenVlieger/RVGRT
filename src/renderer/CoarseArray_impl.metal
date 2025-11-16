// src/renderer/CoarseArray_impl.metal

#include <metal_stdlib>
#include "cumath.h"
#include "CoarseArray.h"
#include "raytracing_functions.h"
#include "TerrainGeneration.h"
using namespace metal;

// --- SDF Kernels ---

kernel void CoarseArray_computeDistX(device const uint* fineData [[buffer(0)]],
                                     device uchar* distX [[buffer(1)]],
                                     uint idx [[thread_position_in_grid]])
{
    if (idx >= SDF_BYTESIZE) return;

    uint64_t cz = idx / (SDF_SIZEX * SDF_SIZEY);
    uint64_t temp = idx % (SDF_SIZEX * SDF_SIZEY);
    uint64_t cy = temp / SDF_SIZEX;
    uint64_t cx = temp % SDF_SIZEX;

    if (isCoarseBlockSolid(cx, cy, cz, fineData)) {
        distX[idx] = 0;
        return;
    }
    
    uchar min_d = SDF_MAX_DIST;
    for (uint i = 1; i <= SDF_MAX_DIST; ++i) {
        if (i <= cx && isCoarseBlockSolid(cx - i, cy, cz, fineData)) {
            min_d = i;
            break;
        }
    }
    for (uint i = 1; i < min_d; ++i) {
        if (cx + i < SDF_SIZEX && isCoarseBlockSolid(cx + i, cy, cz, fineData)) {
            min_d = i;
            break;
        }
    }
    distX[idx] = min_d;
}

kernel void CoarseArray_computeDistY(device const uchar* distX [[buffer(0)]],
                                     device uchar* distY [[buffer(1)]],
                                     uint idx [[thread_position_in_grid]])
{
    if (idx >= SDF_BYTESIZE) return;

    uchar current_dx = distX[idx];
    if (current_dx == 0) {
        distY[idx] = 0;
        return;
    }

    uint64_t cz = idx / (SDF_SIZEX * SDF_SIZEY);
    uint64_t temp = idx % (SDF_SIZEX * SDF_SIZEY);
    uint64_t cy = temp / SDF_SIZEX;

    float min_dist_sq = (float)current_dx * (float)current_dx;
    for (uint y_offset = 1; y_offset <= SDF_MAX_DIST; ++y_offset) {
        if (y_offset * y_offset >= min_dist_sq) break;
        if (cy >= y_offset) {
            uint64_t neighbor_idx = idx - y_offset * SDF_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * (float)distX[neighbor_idx] + (float)(y_offset * y_offset);
            min_dist_sq = fmin(min_dist_sq, dist_sq);
        }
        if (cy + y_offset < SDF_SIZEY) {
            uint64_t neighbor_idx = idx + y_offset * SDF_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * (float)distX[neighbor_idx] + (float)(y_offset * y_offset);
            min_dist_sq = fmin(min_dist_sq, dist_sq);
        }
    }
    distY[idx] = (uchar)fmin((float)SDF_MAX_DIST, sqrt(min_dist_sq));
}

kernel void CoarseArray_computeDistZ(device const uchar* distXY [[buffer(0)]],
                                     device uchar* finalCSDF [[buffer(1)]],
                                     uint idx [[thread_position_in_grid]])
{
    if (idx >= SDF_BYTESIZE) return;

    uchar current_dxy = distXY[idx];
    if (current_dxy == 0) {
        finalCSDF[idx] = 0;
        return;
    }
    
    uint64_t cz = idx / (SDF_SIZEX * SDF_SIZEY);
    float min_dist_sq = (float)current_dxy * (float)current_dxy;
    for (uint z_offset = 1; z_offset <= SDF_MAX_DIST; ++z_offset) {
        if (z_offset * z_offset >= min_dist_sq) break;
        if (cz >= z_offset) {
            uint64_t neighbor_idx = idx - z_offset * (SDF_SIZEX * SDF_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * (float)distXY[neighbor_idx] + (float)(z_offset * z_offset);
            min_dist_sq = fmin(min_dist_sq, dist_sq);
        }
        if (cz + z_offset < SDF_SIZEZ) {
            uint64_t neighbor_idx = idx + z_offset * (SDF_SIZEX * SDF_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * (float)distXY[neighbor_idx] + (float)(z_offset * z_offset);
            min_dist_sq = fmin(min_dist_sq, dist_sq);
        }
    }
    finalCSDF[idx] = (uchar)fmin((float)SDF_MAX_DIST, sqrt(min_dist_sq));
}


// --- GI Kernels ---
kernel void CoarseArray_InitialGlobalIlluminate(device uint* GIdata [[buffer(0)]],
                                                device const uint* bits [[buffer(1)]],
                                                device const uchar* csdf [[buffer(2)]],
                                                constant float3& c_sunDir2 [[buffer(3)]],
                                                uint idx [[thread_position_in_grid]])
{
    // This kernel does not use random numbers, so it is unchanged.
    if (idx >= GI_SIZE) return;
    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;
    float3 worldPos = make_float3((cx + 0.5f) * COARSENESSGI, (cy + 0.5f) * COARSENESSGI, (cz + 0.5f) * COARSENESSGI);
    float3 accumulatedColor = make_float3(0.0f);
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.0001f, bits, csdf);
    if (!shadowHit.hit) {
        accumulatedColor = c_sunColor;
    }
    GIdata[idx] = ((uint)(accumulatedColor.x * 255.f) << 0) | ((uint)(accumulatedColor.y * 255.f) << 8) | ((uint)(accumulatedColor.z * 255.f) << 16) | (255u << 24);
}



kernel void CoarseArray_GlobalIlluminate(device uint* GIdata_curr [[buffer(0)]],
                                         device const uint* bits [[buffer(1)]],
                                         device const uchar* csdf [[buffer(2)]],
                                         texture2d<half, access::sample> texturepack [[texture(3)]],
                                         constant float3& c_sunDir2 [[buffer(4)]],
                                         constant uint& frameNumber [[buffer(5)]],
                                         constant uint64_t& offset [[buffer(6)]],
                                         uint idx_local [[thread_position_in_grid]])
{
    uint64_t idx = idx_local + offset;
    if (idx >= GI_SIZE) return;

    uint random_state = init_random_state(idx, frameNumber);

    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;
    float3 worldPos = make_float3((cx + 0.5f) * COARSENESSGI, (cy + 0.5f) * COARSENESSGI, (cz + 0.5f) * COARSENESSGI);

    if (IsSolid(int3(floor(worldPos)), bits)) { return; }

    float3 newSample = make_float3(0.0f);

    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.001f, bits, csdf);
    if (!shadowHit.hit) {
        newSample += c_sunColor;
    }

    float3 randomDir = random_direction_in_sphere(random_state);
    hitInfo bounceHit = trace(worldPos, randomDir, 0.001f, bits, csdf);

    // (rest of kernel logic is unchanged but will now work correctly)
    if (bounceHit.hit) {
        int3 g = int3(floor(bounceHit.pos / (float)COARSENESSGI));
        if (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < GI_SIZEX && g.y < GI_SIZEY && g.z < GI_SIZEZ) {
            uint64_t hit_idx = (uint64_t)g.z * GI_SIZEX * GI_SIZEY + (uint64_t)g.y * GI_SIZEX + g.x;
            uint prevSample = GIdata_curr[hit_idx];
            float3 bouncedColor = make_float3((prevSample & 255) / 255.0f, ((prevSample >> 8) & 255) / 255.0f, ((prevSample >> 16) & 255) / 255.0f);
            float3 surfaceAlbedo = sampleTexture(bounceHit.uv, bounceHit.pos, texturepack);
            newSample += (bouncedColor * surfaceAlbedo);
        }
    } else {
        newSample += sampleSky(randomDir, c_sunDir2);
    }

    const float LEARNING_RATE = 0.04f;
    uint prevData = GIdata_curr[idx];
    float3 previousColor = make_float3((prevData & 255) / 255.0f, ((prevData >> 8) & 255) / 255.0f, ((prevData >> 16) & 255) / 255.0f);
    float3 finalColor = lerp(previousColor, newSample, LEARNING_RATE);
    finalColor = fmin(finalColor, make_float3(2.0f));

    GIdata_curr[idx] = ((uint)(fmin(finalColor.x, 1.0f) * 255.f) << 0) | ((uint)(fmin(finalColor.y, 1.0f) * 255.f) << 8) | ((uint)(fmin(finalColor.z, 1.0f) * 255.f) << 16) | (255u << 24);
}