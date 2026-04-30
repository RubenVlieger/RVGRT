// Compute Exposure (Log-Average Luminance)
// Computes average scene luminance using parallel reduction.

#include "ShaderTypes.slang"
#include "Settings.slang"

RWStructuredBuffer<ExposureData> exposure;
uniform FrameData frame;

Texture2D<float4> texDirect;
Texture2D<float4> texAccum;
Texture2D<float4> texAlbedo;

static float getLuminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

static groupshared float sharedLogLum[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeExposure(uint3 gid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID)
{
    uint2 dim;
    texDirect.GetDimensions(dim.x, dim.y);

    uint pixelCount = 0;
    float localLogSum = 0.0f;
    
    // Strided sampling: Each thread samples a grid across the screen
    const uint strideX = 32;
    const uint strideY = 32;
    
    for (uint y = tid.y * strideY; y < dim.y; y += 16 * strideY) {
        for (uint x = tid.x * strideX; x < dim.x; x += 16 * strideX) {
            uint2 coords = uint2(x, y);
            if (coords.x >= dim.x || coords.y >= dim.y) continue;

            float3 direct = texDirect[coords].rgb;
            float3 indirect = texAccum[coords].rgb;
            float3 albedo = texAlbedo[coords].rgb;
            
            #if !INDIRECT_LIGHTING
            float3 color = direct * albedo;
            #else
            float3 color = (direct + indirect) * albedo;
            #endif
            float lum = getLuminance(color);
            
            // Center weighting
            float2 uv = float2(x / float(dim.x), y / float(dim.y));
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
    
    GroupMemoryBarrierWithGroupSync();
    
    // Parallel Reduction (256 -> 1)
    if (linearTid < 128) sharedLogLum[linearTid] += sharedLogLum[linearTid + 128];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 64) sharedLogLum[linearTid] += sharedLogLum[linearTid + 64];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 32) sharedLogLum[linearTid] += sharedLogLum[linearTid + 32];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 16) sharedLogLum[linearTid] += sharedLogLum[linearTid + 16];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 8) sharedLogLum[linearTid] += sharedLogLum[linearTid + 8];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 4) sharedLogLum[linearTid] += sharedLogLum[linearTid + 4];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 2) sharedLogLum[linearTid] += sharedLogLum[linearTid + 2];
    GroupMemoryBarrierWithGroupSync();
    if (linearTid < 1) sharedLogLum[linearTid] += sharedLogLum[linearTid + 1];
    GroupMemoryBarrierWithGroupSync();
    
    // Thread 0 writes result
    if (linearTid == 0) {
        float avgLogLum = sharedLogLum[0] / 256.0f;
        float currentSceneLum = exp(avgLogLum);
        currentSceneLum = clamp(currentSceneLum, 0.01f, 60.0f);

        float lastLum = exposure[0].sceneLuminance;
        float adaptationSpeed = (currentSceneLum > lastLum) ? 4.0f : 1.0f;
        float interpolatedLum = lastLum + (currentSceneLum - lastLum) * 
                               (1.0f - exp(-frame.deltaTime * adaptationSpeed));
        
        if (!(interpolatedLum >= 0.001f && interpolatedLum <= 100.0f)) interpolatedLum = 0.5f;
        
        // Add minimum luminance floor as safety net against pure black convergence
        interpolatedLum = max(interpolatedLum, 0.05f);

        exposure[0].sceneLuminance = interpolatedLum;
    }
}
