#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"

// ============================================================================
// KERNEL: Temporal Accumulation
// 
// Reprojects and blends with history buffer for temporal anti-aliasing.
// Uses neighborhood clamping to reject invalid history.
// ============================================================================

KERNEL(TemporalAccumulation)(
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texAccum, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texRawIndirect, 1),
    PARAM_TEXTURE_READ(tex2d_f32_s, texHistory, 2),
    PARAM_TEXTURE_READ(tex2d_f32_r, texMotion, 3),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 4),
    PARAM_TEXTURE_READ(tex2d_f32_r, texPrevDepth, 5),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDirect, 6),
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texAccum);
    int width = texAccum.get_width();
    int height = texAccum.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Read current frame color
#if defined(PLATFORM_METAL)
    float3 currentDirect = TEX_READ_2D(texDirect, gid).rgb;
    float3 currentIndirect = TEX_READ_2D(texRawIndirect, gid).rgb;
#else
    float4 direct4 = TEX_READ_2D(texDirect, gid);
    float4 indirect4 = TEX_READ_2D(texRawIndirect, gid);
    float3 currentDirect = make_float3(direct4.x, direct4.y, direct4.z);
    float3 currentIndirect = make_float3(indirect4.x, indirect4.y, indirect4.z);
#endif
    float3 currentRGB = currentDirect + currentIndirect;
    
    // NaN check
#if defined(PLATFORM_METAL)
    if (any(isnan(currentDirect)) || any(isinf(currentDirect))) {
#else
    if (isnan(currentDirect.x) || isnan(currentDirect.y) || isnan(currentDirect.z) ||
        isinf(currentDirect.x) || isinf(currentDirect.y) || isinf(currentDirect.z)) {
#endif
#if defined(PLATFORM_METAL)
        TEX_WRITE_2D(texAccum, float4(1.0, 0.0, 1.0, 1.0), gid);
#else
        ushort4 errHalf4;
        errHalf4.x = __float2half_rn(1.0f);
        errHalf4.y = __float2half_rn(0.0f);
        errHalf4.z = __float2half_rn(1.0f);
        errHalf4.w = __float2half_rn(1.0f);
        TEX_WRITE_2D_RGBA16F(texAccum, errHalf4, gid);
#endif
        return;
    }
#if defined(PLATFORM_METAL)
    if (any(isnan(currentIndirect)) || any(isinf(currentIndirect))) {
#else
    if (isnan(currentIndirect.x) || isnan(currentIndirect.y) || isnan(currentIndirect.z) ||
        isinf(currentIndirect.x) || isinf(currentIndirect.y) || isinf(currentIndirect.z)) {
#endif
#if defined(PLATFORM_METAL)
        TEX_WRITE_2D(texAccum, float4(0.0, 1.0, 1.0, 1.0), gid);
#else
        ushort4 errHalf4;
        errHalf4.x = __float2half_rn(0.0f);
        errHalf4.y = __float2half_rn(1.0f);
        errHalf4.z = __float2half_rn(1.0f);
        errHalf4.w = __float2half_rn(1.0f);
        TEX_WRITE_2D_RGBA16F(texAccum, errHalf4, gid);
#endif
        return;
    }

    // Motion and UVs
#if defined(PLATFORM_METAL)
    float2 motion = TEX_READ_2D(texMotion, gid).xy;
#else
    float4 motion4 = TEX_READ_2D(texMotion, gid);
    float2 motion = make_float2(motion4.x, motion4.y);
#endif
    float velMag = length(motion);
    float movementFactor = fminf(fmaxf(velMag * 200.0f, 0.0f), 1.0f);

#if defined(PLATFORM_METAL)
    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
#else
    float2 uv = make_float2((gid.x + 0.5f) / (float)width, (gid.y + 0.5f) / (float)height);
#endif
    float2 prevUV = uv - motion;

    // Neighborhood statistics
#if defined(PLATFORM_METAL)
    float3 m1 = float3(0.0f);
    float3 m2 = float3(0.0f);
#else
    float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
#endif
    
    for(int y = -1; y <= 1; ++y) {
        for(int x = -1; x <= 1; ++x) {
#if defined(PLATFORM_METAL)
            uint2 tapCoord = uint2(gid.x + x, gid.y + y);
            tapCoord.x = clamp(tapCoord.x, 0u, (uint)width - 1);
            tapCoord.y = clamp(tapCoord.y, 0u, (uint)height - 1);
#else
            int2 tapCoord = make_int2(gid.x + x, gid.y + y);
            tapCoord.x = max(0, min(tapCoord.x, width - 1));
            tapCoord.y = max(0, min(tapCoord.y, height - 1));
#endif

#if defined(PLATFORM_METAL)
            float3 neighborDirect = TEX_READ_2D(texDirect, tapCoord).rgb;
            float3 neighborIndirect = TEX_READ_2D(texRawIndirect, tapCoord).rgb;
#else
            float4 nd4 = TEX_READ_2D(texDirect, tapCoord);
            float4 ni4 = TEX_READ_2D(texRawIndirect, tapCoord);
            float3 neighborDirect = make_float3(nd4.x, nd4.y, nd4.z);
            float3 neighborIndirect = make_float3(ni4.x, ni4.y, ni4.z);
#endif
            float3 neighborRGB = neighborDirect + neighborIndirect;
            
            // RGB to YCoCg
#if defined(PLATFORM_METAL)
            float Y  = dot(neighborRGB, float3(0.25f, 0.50f, 0.25f));
            float Co = dot(neighborRGB, float3(0.50f, 0.00f, -0.50f));
            float Cg = dot(neighborRGB, float3(-0.25f, 0.50f, -0.25f));
            float3 neighborYCoCg = float3(Y, Co, Cg);
#else
            float Y  = neighborRGB.x * 0.25f + neighborRGB.y * 0.50f + neighborRGB.z * 0.25f;
            float Co = neighborRGB.x * 0.50f + neighborRGB.y * 0.00f - neighborRGB.z * 0.50f;
            float Cg = -neighborRGB.x * 0.25f + neighborRGB.y * 0.50f - neighborRGB.z * 0.25f;
            float3 neighborYCoCg = make_float3(Y, Co, Cg);
#endif

            m1 = m1 + neighborYCoCg;
            m2 = m2 + neighborYCoCg * neighborYCoCg;
        }
    }

    float3 mu = m1 / 9.0f;
    float3 diff = m2 / 9.0f - mu * mu;
    float3 sigma = make_float3(sqrtf(fabsf(diff.x)), sqrtf(fabsf(diff.y)), sqrtf(fabsf(diff.z)));

    float gamma = 10.0f - (10.0f - 0.75f) * movementFactor;
    float3 minColor = mu - gamma * sigma;
    float3 maxColor = mu + gamma * sigma;

    // Sample history
    DECLARE_SAMPLER(sLinear, linear, clamp_to_edge);
#if defined(PLATFORM_METAL)
    float3 historyRGB = TEX_SAMPLE_2D(texHistory, prevUV).rgb;
#else
    float4 hist4 = TEX_SAMPLE_2D(texHistory, prevUV);
    float3 historyRGB = make_float3(hist4.x, hist4.y, hist4.z);
#endif
    if (isnan(historyRGB.x) || isnan(historyRGB.y) || isnan(historyRGB.z))
        historyRGB = currentRGB;

    // YCoCg conversion
#if defined(PLATFORM_METAL)
    float hY  = dot(historyRGB, float3(0.25f, 0.50f, 0.25f));
    float hCo = dot(historyRGB, float3(0.50f, 0.00f, -0.50f));
    float hCg = dot(historyRGB, float3(-0.25f, 0.50f, -0.25f));
    float3 historyYCoCg = float3(hY, hCo, hCg);
#else
    float hY  = historyRGB.x * 0.25f + historyRGB.y * 0.50f + historyRGB.z * 0.25f;
    float hCo = historyRGB.x * 0.50f + historyRGB.y * 0.00f - historyRGB.z * 0.50f;
    float hCg = -historyRGB.x * 0.25f + historyRGB.y * 0.50f - historyRGB.z * 0.25f;
    float3 historyYCoCg = make_float3(hY, hCo, hCg);
#endif

    float3 clampedHistoryYCoCg;
    clampedHistoryYCoCg.x = fmaxf(minColor.x, fminf(historyYCoCg.x, maxColor.x));
    clampedHistoryYCoCg.y = fmaxf(minColor.y, fminf(historyYCoCg.y, maxColor.y));
    clampedHistoryYCoCg.z = fmaxf(minColor.z, fminf(historyYCoCg.z, maxColor.z));

    // YCoCg to RGB
    float3 clampedHistoryRGB;
    float cY = clampedHistoryYCoCg.x;
    float cCo = clampedHistoryYCoCg.y;
    float cCg = clampedHistoryYCoCg.z;
    clampedHistoryRGB.x = cY + cCo - cCg;
    clampedHistoryRGB.y = cY + cCg;
    clampedHistoryRGB.z = cY - cCo - cCg;

    float blendWeight = 0.98f - (0.98f - 0.9f) * movementFactor;
    
    // Depth rejection
    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && 
                        prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    if (validHistory) {
#if defined(PLATFORM_METAL)
        uint2 prevCoords = uint2(prevUV.x * width, prevUV.y * height);
#else
        uint2 prevCoords = make_uint2((unsigned int)(prevUV.x * width), (unsigned int)(prevUV.y * height));
#endif
#if defined(PLATFORM_METAL)
        float currentDepth = TEX_READ_2D(texDepth, gid).r;
        float prevDepth = TEX_READ_2D(texPrevDepth, prevCoords).r;
#else
        float4 cd4 = TEX_READ_2D(texDepth, gid);
        float4 pd4 = TEX_READ_2D(texPrevDepth, prevCoords);
        float currentDepth = cd4.x;
        float prevDepth = pd4.x;
#endif
        
        float diff = fabsf(currentDepth - prevDepth) / (currentDepth + 1e-5f);
        if (diff > 0.05f) {
            blendWeight = 0.0f;
        }
    } else {
        blendWeight = 0.0f;
    }

    // Blend
    float3 result = currentRGB + (clampedHistoryRGB - currentRGB) * blendWeight;
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texAccum, float4(result, 1.0f), gid);
#else
    ushort4 resultHalf4;
    resultHalf4.x = __float2half_rn(result.x);
    resultHalf4.y = __float2half_rn(result.y);
    resultHalf4.z = __float2half_rn(result.z);
    resultHalf4.w = __float2half_rn(1.0f);
    TEX_WRITE_2D_RGBA16F(texAccum, resultHalf4, gid);
#endif
}
