#include <metal_stdlib>
#include "cumath.h"
#include "shader_macros.h"
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"

#if defined(PLATFORM_METAL)
using namespace metal;
#endif

// ============================================================================
// KERNEL: Temporal Accumulation
// 
// Reprojects and blends with history buffer for temporal anti-aliasing.
// Uses neighborhood clamping to reject invalid history.
// ============================================================================

inline float3 RGBToYCoCg(float3 rgb) {
    float Y  = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));
    float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    return float3(Y, Co, Cg);
}

inline float3 YCoCgToRGB(float3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

KERNEL(TemporalAccumulation)(
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texAccum, 0),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texRawIndirect, 1),
    PARAM_TEXTURE_READ(texture2d<float, access::sample>, texHistory, 2),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texMotion, 3),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDepth, 4),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texPrevDepth, 5),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDirect, 6),
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texAccum);
    uint2 gid = GET_GID();
    int width = texAccum.get_width();
    int height = texAccum.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Read current frame color
    float3 currentDirect = TEX_READ_2D(texDirect, gid).rgb;
    float3 currentIndirect = TEX_READ_2D(texRawIndirect, gid).rgb;
    float3 currentRGB = currentDirect + currentIndirect;
    
    // NaN check
    if (any(isnan(currentDirect)) || any(isinf(currentDirect))) {
        TEX_WRITE_2D(texAccum, float4(1.0, 0.0, 1.0, 1.0), gid);
        return;
    }
    if (any(isnan(currentIndirect)) || any(isinf(currentIndirect))) {
        TEX_WRITE_2D(texAccum, float4(0.0, 1.0, 1.0, 1.0), gid);
        return;
    }

    // Motion and UVs
    float2 motion = TEX_READ_2D(texMotion, gid).xy;
    float velMag = length(motion);
    float movementFactor = saturate(velMag * 200.0f);

    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 prevUV = uv - motion;

    // Neighborhood statistics
    float3 m1 = float3(0.0f);
    float3 m2 = float3(0.0f);
    
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

            float3 neighborRGB = TEX_READ_2D(texDirect, tapCoord).rgb + 
                                TEX_READ_2D(texRawIndirect, tapCoord).rgb;
            float3 neighborYCoCg = RGBToYCoCg(neighborRGB);

            m1 += neighborYCoCg;
            m2 += neighborYCoCg * neighborYCoCg;
        }
    }

    float3 mu = m1 / 9.0f;
    float3 sigma = sqrt(abs(m2 / 9.0f - mu * mu));

    float gamma = mix(10.0f, 0.75f, movementFactor);
    float3 minColor = mu - gamma * sigma;
    float3 maxColor = mu + gamma * sigma;

    // Sample history
    DECLARE_SAMPLER(sLinear, linear, clampededge);
    float3 historyRGB = TEX_SAMPLE_2D(texHistory, prevUV).rgb;
    if (isnan(historyRGB.x) || isnan(historyRGB.y) || isnan(historyRGB.z))
        historyRGB = currentRGB;

    float3 historyYCoCg = RGBToYCoCg(historyRGB);
    float3 clampedHistoryYCoCg = clamp(historyYCoCg, minColor, maxColor);
    float3 clampedHistoryRGB = YCoCgToRGB(clampedHistoryYCoCg);

    float blendWeight = mix(0.98f, 0.9f, movementFactor);
    
    // Depth rejection
    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && 
                        prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    if (validHistory) {
#if defined(PLATFORM_METAL)
        uint2 prevCoords = uint2(prevUV.x * width, prevUV.y * height);
#else
        int2 prevCoords = make_int2(prevUV.x * width, prevUV.y * height);
#endif
        float currentDepth = TEX_READ_2D(texDepth, gid).r;
        float prevDepth = TEX_READ_2D(texPrevDepth, prevCoords).r;
        
        float diff = abs(currentDepth - prevDepth) / (currentDepth + 1e-5f);
        if (diff > 0.05f) {
            blendWeight = 0.0f;
        }
    } else {
        blendWeight = 0.0f;
    }

    // Blend
    float3 result = mix(currentRGB, clampedHistoryRGB, blendWeight);
    TEX_WRITE_2D(texAccum, float4(result, 1.0f), gid);
}
