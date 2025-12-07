#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;


// =================================================================================
// KERNEL: A-TROUS EDGE-AVOIDING FILTER
// =================================================================================
kernel void BilateralDenoise(
    texture2d<float, access::write> texDenoised [[texture(0)]],
    texture2d<float, access::read>  texAccum    [[texture(1)]],
    texture2d<float, access::read>  texNormal   [[texture(2)]],
    texture2d<float, access::read>  texDepth    [[texture(3)]],
    constant int& step_width        [[buffer(0)]], 
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texDenoised.get_width() || gid.y >= texDenoised.get_height()) return;

    // 1. Center Tap Data
    float3 centerC = texAccum.read(gid).rgb;
    float3 centerN = texNormal.read(gid).rgb;
    float centerD  = texDepth.read(gid).r;

    // Gaussian-approximate weights for 3x3
    const float kernelWeights[3] = { 1.0f, 2.0f / 1.0f, 4.0f / 1.0f };

    float3 sumColor = float3(0.0f);
    float sumWeight = 0.0f;

    // 3. Iteration (3x3 grid with holes)
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            
            // Offset coordinate by step_width
            int2 offset = int2(x, y) * step_width;
            uint2 tapCoord = uint2(gid.x + offset.x, gid.y + offset.y);

            // Bounds check
            if(tapCoord.x >= texDenoised.get_width() || tapCoord.y >= texDenoised.get_height()) {
                tapCoord = gid;
            }

            float3 tapC = texAccum.read(tapCoord).rgb;
            float3 tapN = texNormal.read(tapCoord).rgb;
            float tapD  = texDepth.read(tapCoord).r;

            // --- A. Normal Weight
            float dotN = max(dot(centerN, tapN), 0.0f);
            float wNormal = pow(dotN, 16.0f); // High power ensures we don't bleed colors around voxel corners

            // --- B. Depth Weight (Plane Distance) ---
            // 1.0 = Allow 1 unit (1 block) of depth deviation before rejecting
            float wDepth = (abs(centerD - tapD) < 1.5f) ? 1.0f : 0.0f;
            

            // Calculate Kernel Weight (Gaussian)
            float kWeight = kernelWeights[abs(x)] * kernelWeights[abs(y)];
            // Combine
            float w = wNormal * wDepth * kWeight;

            sumColor  += tapC * w;
            sumWeight += w;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = centerC;
        sumWeight = 1.0f;
    }
    texDenoised.write(float4(sumColor / sumWeight, 1.0f), gid);
}
