#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;



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



// =================================================================================
// KERNEL: ADVANCED TEMPORAL ACCUMULATION
// =================================================================================
kernel void TemporalAccumulation(
    texture2d<float, access::write> texAccum      [[texture(0)]],
    texture2d<float, access::read>  texRawIndirect[[texture(1)]],
    texture2d<float, access::sample> texHistory   [[texture(2)]], 
    texture2d<float, access::read>  texMotion     [[texture(3)]],
    texture2d<float, access::read>  texDepth      [[texture(4)]],
    texture2d<float, access::read>  texPrevDepth  [[texture(5)]], 
    texture2d<float, access::read>  texDirect     [[texture(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texAccum.get_width() || gid.y >= texAccum.get_height()) return;

    // 1. Read Current Frame Color (Direct + Indirect)
    float3 currentDirect = texDirect.read(gid).rgb;
    float3 currentIndirect = texRawIndirect.read(gid).rgb;
    float3 currentRGB = currentDirect + currentIndirect;
    
    if (any(isnan(currentDirect)) || any(isinf(currentDirect))) {
        texAccum.write(float4(1.0, 0.0, 1.0, 1.0), gid); // Magenta for direct NaN
        return;
    }
    if (any(isnan(currentIndirect)) || any(isinf(currentIndirect))) {
        texAccum.write(float4(0.0, 1.0, 1.0, 1.0), gid); // Cyan for indirect NaN
        return;
    }    
    // 2. Motion and UVs
    float2 motion = texMotion.read(gid).xy;
    float velMag = length(motion);
    float movementFactor = saturate(velMag * 200.0f); 
    

    float2 uv = (float2(gid) + 0.5f) / float2(texAccum.get_width(), texAccum.get_height());
    float2 prevUV = uv - motion;

    // 3. Neighborhood Statistics (Variance Calculation)
    float3 m1 = float3(0.0f); // First moment (Mean)
    float3 m2 = float3(0.0f); // Second moment (Variance)
    
    // We sample a 3x3 neighborhood
    for(int y = -1; y <= 1; ++y) {
        for(int x = -1; x <= 1; ++x) {
            uint2 tapCoord = uint2(gid.x + x, gid.y + y);
            
            // Boundary checks (clamp to edge)
            tapCoord.x = clamp(tapCoord.x, 0u, texAccum.get_width() - 1);
            tapCoord.y = clamp(tapCoord.y, 0u, texAccum.get_height() - 1);

            float3 neighborRGB = texDirect.read(tapCoord).rgb + texRawIndirect.read(tapCoord).rgb;
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

    // 5. Sample History
    constexpr sampler sLinear(filter::linear);

    float3 historyRGB = texHistory.sample(sLinear, prevUV).rgb;
    if (isnan(historyRGB.x) || isnan(historyRGB.y) || isnan(historyRGB.z))
        historyRGB = currentRGB;

    //float3 historyRGB = texHistory.sample(sLinear, prevUV).rgb;
    float3 historyYCoCg = RGBToYCoCg(historyRGB);

    // 6. CLIP History to Box
    // Instead of hard clamp, we clip the vector towards the center (better color stability)
    // But for performance/simplicity, hard clamping in YCoCg is usually sufficient.
    float3 clampedHistoryYCoCg = clamp(historyYCoCg, minColor, maxColor);
    float3 clampedHistoryRGB = YCoCgToRGB(clampedHistoryYCoCg);


    float blendWeight = mix(0.98f, 0.9f, movementFactor);
    
    // 8. Depth Rejection (Disocclusion Check)
    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    if (validHistory) {
        uint2 prevCoords = uint2(prevUV.x * texPrevDepth.get_width(), prevUV.y * texPrevDepth.get_height());
        float currentDepth = texDepth.read(gid).r;
        float prevDepth = texPrevDepth.read(prevCoords).r;
        
        // Use relative difference
        float diff = abs(currentDepth - prevDepth) / (currentDepth + 1e-5f);
        if (diff > 0.05f) { // Stricter threshold
            blendWeight = 0.0f; // Reset
        }
    } else {
        blendWeight = 0.0f;
    }

    // 8. Blend
    float3 result = mix(currentRGB, clampedHistoryRGB, blendWeight);
    texAccum.write(float4(result, 1.0f), gid);
}
