#include <metal_stdlib>
#include "cumath.h"
#include "shader_macros.h"
#include "renderer/ShaderTypes.h"

#if defined(PLATFORM_METAL)
using namespace metal;
#endif

// ============================================================================
// KERNEL: Compute Exposure (Log-Average Luminance)
// 
// Computes average scene luminance using parallel reduction.
// Used for auto-exposure tone mapping.
// ============================================================================

inline float getLuminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

KERNEL(ComputeExposure)(
    PARAM_BUFFER(device ExposureData, exposure, 0),
    PARAM_CONSTANT(FrameData, frame, 1),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDirect, 0),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texAccum, 1),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texAlbedo, 2),
    DECLARE_GID(),
    DECLARE_TID()
)
{
#if defined(PLATFORM_METAL)
    uint width = texDirect.get_width();
    uint height = texDirect.get_height();
    uint2 gid = GET_GID();
    uint2 tid = GET_TID();
#else
    int2 gid = GET_GID();
    int2 tid = make_int2(threadIdx.x, threadIdx.y);
    int width = _width;
    int height = _height;
#endif

    // Shared memory for reduction
    SHARED_MEM(float, sharedLogLum, 256);
    
    uint pixelCount = 0;
    float localLogSum = 0.0f;
    
    // Strided sampling: Each thread samples a grid across the screen
    const uint strideX = 32;
    const uint strideY = 32;
    
#if defined(PLATFORM_METAL)
    for (uint y = tid.y * strideY; y < height; y += 16 * strideY) {
        for (uint x = tid.x * strideX; x < width; x += 16 * strideX) {
            uint2 coords = uint2(x, y);
            if (coords.x >= width || coords.y >= height) continue;
#else
    for (int y = tid.y * strideY; y < height; y += 16 * strideY) {
        for (int x = tid.x * strideX; x < width; x += 16 * strideX) {
            int2 coords = make_int2(x, y);
            if (coords.x >= width || coords.y >= height) continue;
#endif

            float3 direct = TEX_READ_2D(texDirect, coords).rgb;
            float3 indirect = TEX_READ_2D(texAccum, coords).rgb;
            float3 albedo = TEX_READ_2D(texAlbedo, coords).rgb;
            
            float3 color = (direct + indirect) * albedo;
            float lum = getLuminance(color);
            
            // Center weighting
            float2 uv = float2(x, y) / float2(width, height);
            float dist = length(uv - 0.5f);
            float weight = 1.0f - smoothstep(0.2f, 0.6f, dist);
            weight = max(weight, 0.1f);
            
            localLogSum += log(max(lum, 0.0001f)) * weight;
            pixelCount++;
        }
    }
    
    // Store in shared memory
    uint linearTid = tid.y * 16 + tid.x;
    sharedLogLum[linearTid] = (pixelCount > 0) ? (localLogSum / float(pixelCount)) : -9.0f;
    
    BARRIER_GROUP();
    
    // Parallel Reduction (256 -> 1)
    if (linearTid < 128) sharedLogLum[linearTid] += sharedLogLum[linearTid + 128];
    BARRIER_GROUP();
    if (linearTid < 64) sharedLogLum[linearTid] += sharedLogLum[linearTid + 64];
    BARRIER_GROUP();
    if (linearTid < 32) sharedLogLum[linearTid] += sharedLogLum[linearTid + 32];
    BARRIER_GROUP();
    if (linearTid < 16) sharedLogLum[linearTid] += sharedLogLum[linearTid + 16];
    BARRIER_GROUP();
    if (linearTid < 8) sharedLogLum[linearTid] += sharedLogLum[linearTid + 8];
    BARRIER_GROUP();
    if (linearTid < 4) sharedLogLum[linearTid] += sharedLogLum[linearTid + 4];
    BARRIER_GROUP();
    if (linearTid < 2) sharedLogLum[linearTid] += sharedLogLum[linearTid + 2];
    BARRIER_GROUP();
    if (linearTid < 1) sharedLogLum[linearTid] += sharedLogLum[linearTid + 1];
    BARRIER_GROUP();
    
    // Thread 0 writes result
    if (linearTid == 0) {
        float avgLogLum = sharedLogLum[0] / 256.0f;
        float currentSceneLum = exp(avgLogLum);
        currentSceneLum = clamp(currentSceneLum, 0.01f, 60.0f);

        float lastLum = exposure.sceneLuminance;
        float adaptationSpeed = (currentSceneLum > lastLum) ? 4.0f : 1.0f;
        float interpolatedLum = lastLum + (currentSceneLum - lastLum) * 
                               (1.0f - exp(-frame.deltaTime * adaptationSpeed));
        
        if (isnan(interpolatedLum)) interpolatedLum = 0.5f;

        exposure.sceneLuminance = interpolatedLum;
    }
}
