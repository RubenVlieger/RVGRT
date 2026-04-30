// Temporal Accumulation
// Reprojects and blends with history buffer for temporal anti-aliasing.

#include "Settings.slang"

RWTexture2D<float4> texAccum;
Texture2D<float4> texRawIndirect;
Texture2D<float4> texHistory;
Texture2D<float4> texMotion;
Texture2D<float4> texDepth;
Texture2D<float4> texPrevDepth;
Texture2D<float4> texDirect;

SamplerState historySampler;

static float3 RGBToYCoCg(float3 rgb) {
    float Y  = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));
    float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    return float3(Y, Co, Cg);
}

static float3 YCoCgToRGB(float3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void TemporalAccumulation(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texAccum.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    #if !INDIRECT_LIGHTING
    texAccum[gid.xy] = float4(0.0f);
    return;
    #endif

    // Read current frame color
    float3 currentIndirect = texRawIndirect[gid.xy].rgb;
    
    // NaN check
    // Fast-math proof NaN check: a NaN value is the only value that doesn't equal itself
    if (!(currentIndirect.x >= -1e30f && currentIndirect.x <= 1e30f) || 
        !(currentIndirect.y >= -1e30f && currentIndirect.y <= 1e30f) || 
        !(currentIndirect.z >= -1e30f && currentIndirect.z <= 1e30f)) {
        texAccum[gid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Motion and UVs
    float2 motion = texMotion[gid.xy].xy;
    float velMag = length(motion);
    float movementFactor = saturate(velMag * 200.0f);

    float2 uv = (float2(gid.xy) + 0.5f) / float2(dim);
    float2 prevUV = uv - motion;

    // Neighborhood statistics
    float3 m1 = float3(0.0f);
    float3 m2 = float3(0.0f);
    
    for(int y = -1; y <= 1; ++y) {
        for(int x = -1; x <= 1; ++x) {
            uint2 tapCoord = uint2(gid.x + x, gid.y + y);
            tapCoord.x = clamp(tapCoord.x, 0u, dim.x - 1);
            tapCoord.y = clamp(tapCoord.y, 0u, dim.y - 1);

            float3 neighborRGB = texRawIndirect[tapCoord].rgb;
            float3 neighborYCoCg = RGBToYCoCg(neighborRGB);

            m1 += neighborYCoCg;
            m2 += neighborYCoCg * neighborYCoCg;
        }
    }

    float3 mu = m1 / 9.0f;
    float3 sigma = sqrt(abs(m2 / 9.0f - mu * mu));

    float gamma = lerp(10.0f, 0.75f, movementFactor);
    float3 minColor = mu - gamma * sigma;
    float3 maxColor = mu + gamma * sigma;

    // Sample history
    float3 historyRGB = texHistory.Sample(historySampler, prevUV).rgb;
    if (!(historyRGB.x >= 0.0f || historyRGB.x < 0.0f) || 
        !(historyRGB.y >= 0.0f || historyRGB.y < 0.0f) || 
        !(historyRGB.z >= 0.0f || historyRGB.z < 0.0f))
        historyRGB = currentIndirect;

    float3 historyYCoCg = RGBToYCoCg(historyRGB);
    float3 clampedHistoryYCoCg = clamp(historyYCoCg, minColor, maxColor);
    float3 clampedHistoryRGB = YCoCgToRGB(clampedHistoryYCoCg);

    float blendWeight = lerp(0.98f, 0.9f, movementFactor);
    
    // Depth rejection
    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && 
                        prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    if (validHistory) {
        uint2 prevCoords = uint2(uint(prevUV.x * dim.x), uint(prevUV.y * dim.y));
        float currentDepth = texDepth[gid.xy].r;
        float prevDepth = texPrevDepth[prevCoords].r;
        
        float diff = abs(currentDepth - prevDepth) / (currentDepth + 1e-5f);
        if (diff > 0.05f) {
            blendWeight = 0.0f;
        }
    } else {
        blendWeight = 0.0f;
    }

    // Blend
    float3 result = lerp(currentIndirect, clampedHistoryRGB, blendWeight);
    texAccum[gid.xy] = float4(result, 1.0f);
}
