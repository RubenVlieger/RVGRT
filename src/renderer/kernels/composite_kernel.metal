#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
#include "renderer/shader_settings.h"

using namespace metal;

// Standard ACES fitted tone mapper (Unreal Engine 4 version)
// Compresses High Dynamic Range (e.g., 0 to 100) to LDR (0 to 1).
inline float3 ACESFilm(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

inline float3 LinearToSRGB(float3 color) {
    return select(1.055f * pow(color, 1.0f / 2.4f) - 0.055f,
                  12.92f * color,
                  color <= 0.0031308f);
}

inline float3 applyContrast(float3 color, float contrast) {
    return max(float3(0.0f), (color - 0.5f) * contrast + 0.5f);
}

// Saturation boost (Luma-based)
inline float3 applySaturation(float3 color, float saturation) {
    // Standard Luma coefficients (Rec. 709)
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    return mix(float3(luma), color, saturation);
}

float3 SampleFogBilateral(
    texture2d<float, access::sample> texVolumetric, 
    texture2d<float, access::read> texDepth, 
    float2 uv, 
    float currentDepth, 
    float2 screenSize) 
{
    float2 lowResSize = screenSize * 0.5f;
    float2 pixelPos = uv * lowResSize - 0.5f;
    
    int2 basePos = int2(floor(pixelPos));
    float2 weights = fract(pixelPos);
    
    float3 sumColor = float3(0.0f);
    float sumWeight = 0.0f;
    
    // 2. Loop through 2x2 neighbors
    for(int y = 0; y <= 1; y++) {
        for(int x = 0; x <= 1; x++) {
            int2 offset = int2(x, y);
            int2 coord = basePos + offset;
            
            // Clamp
            coord.x = max(0, min(coord.x, int(lowResSize.x) - 1));
            coord.y = max(0, min(coord.y, int(lowResSize.y) - 1));
            
            float3 fogColor = texVolumetric.read(uint2(coord)).rgb;
            
            // 3. READ DEPTH AT THE LOCATION THAT GENERATED THIS FOG
            uint2 depthCoord = uint2(coord) * 2;
            float neighborDepth = texDepth.read(depthCoord).r;
            
            // 4. Calculate Weight based on Depth Similarity
            float depthDiff = abs(currentDepth - neighborDepth);
            float depthWeight = 1.0f / (1.0f + depthDiff * 2.0f);
            
            // Bilinear weight (standard upscaling)
            float bilinearWeight = (x == 0 ? (1.0f - weights.x) : weights.x) * 
                                   (y == 0 ? (1.0f - weights.y) : weights.y);
            
            float combinedWeight = depthWeight * bilinearWeight;
            
            sumColor += fogColor * combinedWeight;
            sumWeight += combinedWeight;
        }
    }
    return sumColor / (sumWeight + 0.0001f);
}



// =================================================================================
// KERNEL: COMPOSITE 
// =================================================================================
kernel void Composite(
    texture2d<float, access::write> texFinal   [[texture(0)]],
    texture2d<float, access::read>  texDirect  [[texture(1)]],
    texture2d<float, access::read>  texAccum   [[texture(2)]], 
    texture2d<float, access::read>  texAlbedo  [[texture(3)]],
    texture2d<float, access::read>  texDepth   [[texture(4)]],
    texture2d<float, access::sample> texVolumetric [[texture(5)]],
    
    device ExposureData& exposure [[buffer(0)]],

    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texFinal.get_width() || gid.y >= texFinal.get_height()) return;

    // 1. Gather Data
    float3 directLight   = texDirect.read(gid).rgb;
    float3 indirectLight = texAccum.read(gid).rgb;
    float3 albedo        = texAlbedo.read(gid).rgb;
    float depth          = texDepth.read(gid).r;
    
    constexpr sampler sLinear(filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(texFinal.get_width(), texFinal.get_height());
    float2 screenSize = float2(texFinal.get_width(), texFinal.get_height());
    float3 fog = SampleFogBilateral(texVolumetric, texDepth, uv, depth, screenSize);

    // 3. Apply Material (Linear Space)
    float3 totalIrradiance = directLight + indirectLight;
    float3 linearColor = totalIrradiance * albedo + fog;

    // 4. Fog Logic (unchanged)
    if (depth < 50000.0f) 
    {
        const float fogStart = COMPOSITE_FOG_START;
        const float fogDensity = COMPOSITE_FOG_DENSITY; 
        float dist = max(depth - fogStart, 0.0f);
        float fogFactor = 1.0f - exp(-dist * fogDensity);
        float3 fogColor = float3(COMPOSITE_FOG_COLOR); 
        linearColor = mix(linearColor, fogColor, fogFactor);
    }

    // 5. COLOR GRADING (UPDATED with Auto-Exposure)
    
    // Retrieve calculated luminance from previous pass
    float avgLum = exposure.sceneLuminance;
    
    // Standard Exposure Formula:
    // Middle Grey (0.18 or 0.5 depending on calibration) / Average Luminance
    // We adjust the key value (1.0f) to taste.
    float exposureScale = 0.15f / (max(avgLum, 0.001f));
    
    linearColor *= exposureScale;

    // Saturation Boost 
    linearColor = applySaturation(linearColor, (depth > 50000.0f) ? SKY_IMAGE_SATURATION : IMAGE_SATURATION); 

    // Tone Mapping (ACES)
    float3 toneMapped = ACESFilm(linearColor);

    // Gamma Correction (Linear -> sRGB)
    float3 finalColor = LinearToSRGB(toneMapped);

    texFinal.write(float4(finalColor, 1.0f), gid);
}
