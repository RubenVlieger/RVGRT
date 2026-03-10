#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "renderer/ShaderTypes.h"

// ============================================================================
// KERNEL: A-Trous Edge-Avoiding Bilateral Filter
// 
// Performs edge-preserving denoising using normal and depth discontinuities.
// Part of the A-Trous wavelet transform with 3 iterations.
// ============================================================================

KERNEL(BilateralDenoise)(
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texDenoised, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texAccum, 1),
    PARAM_TEXTURE_READ(tex2d_f32_r, texNormal, 2),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 3),
    PARAM_CONSTANT(int, step_width, 0),
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texDenoised);
    int width = texDenoised.get_width();
    int height = texDenoised.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Center tap data
#if defined(PLATFORM_METAL)
    float3 centerC = TEX_READ_2D(texAccum, gid).rgb;
    float3 centerN = TEX_READ_2D(texNormal, gid).rgb;
    float centerD = TEX_READ_2D(texDepth, gid).r;
#else
    float4 centerC4 = TEX_READ_2D(texAccum, gid);
    float4 centerN4 = TEX_READ_2D(texNormal, gid);
    float4 centerD4 = TEX_READ_2D(texDepth, gid);
    float3 centerC = make_float3(centerC4.x, centerC4.y, centerC4.z);
    float3 centerN = make_float3(centerN4.x, centerN4.y, centerN4.z);
    float centerD = centerD4.x;
#endif

    // Gaussian-approximate weights for 3x3
    const float kernelWeights[3] = { 1.0f, 2.0f / 1.0f, 4.0f / 1.0f };

    float3 sumColor = float3(0.0f);
    float sumWeight = 0.0f;

    // 3x3 grid with holes (A-Trous)
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            int2 offset = int2(x, y) * step_width;
            
#if defined(PLATFORM_METAL)
            uint2 tapCoord = uint2(gid.x + offset.x, gid.y + offset.y);
            if(tapCoord.x >= (uint)width || tapCoord.y >= (uint)height) {
                tapCoord = gid;
            }
#else
            int2 tapCoord = make_int2(gid.x + offset.x, gid.y + offset.y);
            if(tapCoord.x >= width || tapCoord.y >= height || tapCoord.x < 0 || tapCoord.y < 0) {
                tapCoord = gid;
            }
#endif

#if defined(PLATFORM_METAL)
            float3 tapC = TEX_READ_2D(texAccum, tapCoord).rgb;
            float3 tapN = TEX_READ_2D(texNormal, tapCoord).rgb;
            float tapD = TEX_READ_2D(texDepth, tapCoord).r;
#else
            float4 tapC4 = TEX_READ_2D(texAccum, tapCoord);
            float4 tapN4 = TEX_READ_2D(texNormal, tapCoord);
            float4 tapD4 = TEX_READ_2D(texDepth, tapCoord);
            float3 tapC = make_float3(tapC4.x, tapC4.y, tapC4.z);
            float3 tapN = make_float3(tapN4.x, tapN4.y, tapN4.z);
            float tapD = tapD4.x;
#endif;

            // Normal weight
            float dotN = max(dot(centerN, tapN), 0.0f);
            float wNormal = pow(dotN, 16.0f);

            // Depth weight
            float wDepth = (abs(centerD - tapD) < 1.5f) ? 1.0f : 0.0f;

            // Kernel weight
            float kWeight = kernelWeights[abs(x)] * kernelWeights[abs(y)];
            float w = wNormal * wDepth * kWeight;

            sumColor += tapC * w;
            sumWeight += w;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = centerC;
        sumWeight = 1.0f;
    }
    
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texDenoised, float4(sumColor / sumWeight, 1.0f), gid);
#else
    float3 finalColor = sumColor / sumWeight;
    TEX_WRITE_2D(texDenoised, make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f), gid);
#endif
}
