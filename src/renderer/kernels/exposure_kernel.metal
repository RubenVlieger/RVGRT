#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;




inline float getLuminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

// =================================================================================
// KERNEL: Compute Exposure (Log-Average Luminance)
// =================================================================================
kernel void ComputeExposure(
    device ExposureData& exposure     [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    texture2d<float, access::read> texDirect [[texture(0)]],
    texture2d<float, access::read> texAccum  [[texture(1)]],
    texture2d<float, access::read> texAlbedo [[texture(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    
    // Shared memory for reduction
    threadgroup float sharedLogLum[256];
    
    uint width = texDirect.get_width();
    uint height = texDirect.get_height();
    uint pixelCount = 0;
    float localLogSum = 0.0f;
    
    // Strided sampling: Each thread samples a grid across the screen.
    // We skip pixels to save performance (we don't need every pixel for exposure).
    // Stride 16 gives us enough samples.
    const uint strideX = 32; 
    const uint strideY = 32;
    
    for (uint y = tid.y * strideY; y < height; y += 16 * strideY) {
        for (uint x = tid.x * strideX; x < width; x += 16 * strideX) {
            
            uint2 coords = uint2(x, y);
            if (coords.x >= width || coords.y >= height) continue;

            float3 direct = texDirect.read(coords).rgb;
            float3 indirect = texAccum.read(coords).rgb;
            float3 albedo = texAlbedo.read(coords).rgb;
            
            // Reconstruct approximate linear color (ignoring fog for exposure to keep it focused on geometry)
            float3 color = (direct + indirect) * albedo;
            
            float lum = getLuminance(color);
            
            // Center Weighting: We care more about what's in the center of the screen
            float2 uv = float2(x, y) / float2(width, height);
            float dist = length(uv - 0.5f);
            float weight = 1.0f - smoothstep(0.2f, 0.6f, dist); // Center is 1.0, edges 0.0
            weight = max(weight, 0.1f); // Minimum weight so we don't ignore edges entirely
            
            // Log average (add epsilon to avoid log(0))
            localLogSum += log(max(lum, 0.0001f)) * weight;
            pixelCount++;
        }
    }
    
    // Store in shared memory (Normalize by pixel count immediately to avoid huge numbers)
    // Note: This is a simplified reduction. For perfect weighting we'd need to sum weights too,
    // but assuming uniform distribution of samples, dividing by count is acceptable.
    sharedLogLum[tid.y * 16 + tid.x] = (pixelCount > 0) ? (localLogSum / float(pixelCount)) : -9.0f;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel Reduction in Shared Memory (256 -> 1)
    uint linearTid = tid.y * 16 + tid.x;
    
    if (linearTid < 128) sharedLogLum[linearTid] += sharedLogLum[linearTid + 128]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 64)  sharedLogLum[linearTid] += sharedLogLum[linearTid + 64];  threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 32)  sharedLogLum[linearTid] += sharedLogLum[linearTid + 32];  threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 16)  sharedLogLum[linearTid] += sharedLogLum[linearTid + 16];  threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 8)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 8];   threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 4)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 4];   threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 2)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 2];   threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linearTid < 1)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 1];   threadgroup_barrier(mem_flags::mem_threadgroup);
    

    // Only thread 0 writes the result
    if (linearTid == 0) {
        float avgLogLum = sharedLogLum[0] / 256.0f;
        float currentSceneLum = exp(avgLogLum);
        
        // Clamp extreme values (prevent complete black or infinity)
        currentSceneLum = clamp(currentSceneLum, 0.01f, 60.0f);

        // Temporal Adaptation (Eye Adaptation)
        // Lerp between previous frame luminance and current target
        // Dark to Bright adapts faster than Bright to Dark (usually)
        float lastLum = exposure.sceneLuminance;
        
        float adaptationSpeed = (currentSceneLum > lastLum) ? 4.0f : 1.0f; 
        
        // Time corrected lerp formula: val = mix(curr, target, 1 - exp(-dt * speed))
        float interpolatedLum = lastLum + (currentSceneLum - lastLum) * (1.0f - exp(-frame.deltaTime * adaptationSpeed));
        
        // Safety check for NaN
        if (isnan(interpolatedLum)) interpolatedLum = 0.5f;

        exposure.sceneLuminance = interpolatedLum;
    }
}
