#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
#include "renderer/shader_settings.h"

// ============================================================================
// KERNEL: Composite
// 
// Final composition pass. Combines direct light, indirect light, albedo,
// and volumetric fog. Applies tone mapping and gamma correction.
// ============================================================================

// Standard ACES fitted tone mapper
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

inline float3 applySaturation(float3 color, float saturation) {
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    return mix(float3(luma), color, saturation);
}

float3 SampleFogBilateral(
    PARAM_TEXTURE_READ(texture2d<float, access::sample>, texVolumetric),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDepth),
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
    
    for(int y = 0; y <= 1; y++) {
        for(int x = 0; x <= 1; x++) {
            int2 offset = int2(x, y);
            int2 coord = basePos + offset;
            
            coord.x = max(0, min(coord.x, int(lowResSize.x) - 1));
            coord.y = max(0, min(coord.y, int(lowResSize.y) - 1));
            
#if defined(PLATFORM_METAL)
            float3 fogColor = texVolumetric.read(uint2(coord)).rgb;
            uint2 depthCoord = uint2(coord) * 2;
#else
            float3 fogColor = texVolumetric.read(coord).rgb;
            int2 depthCoord = make_int2(coord.x * 2, coord.y * 2);
#endif
            float neighborDepth = texDepth.read(depthCoord).r;
            
            float depthDiff = abs(currentDepth - neighborDepth);
            float depthWeight = 1.0f / (1.0f + depthDiff * 2.0f);
            
            float bilinearWeight = (x == 0 ? (1.0f - weights.x) : weights.x) * 
                                   (y == 0 ? (1.0f - weights.y) : weights.y);
            
            float combinedWeight = depthWeight * bilinearWeight;
            
            sumColor += fogColor * combinedWeight;
            sumWeight += combinedWeight;
        }
    }
    return sumColor / (sumWeight + 0.0001f);
}

KERNEL(Composite)(
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texFinal, 0),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDirect, 1),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texAccum, 2),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texAlbedo, 3),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDepth, 4),
    PARAM_TEXTURE_READ(texture2d<float, access::sample>, texVolumetric, 5),
    
    PARAM_BUFFER(device ExposureData, exposure, 0),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texFinal);
    uint2 gid = GET_GID();
    int width = texFinal.get_width();
    int height = texFinal.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Gather Data
    float3 directLight = TEX_READ_2D(texDirect, gid).rgb;
    float3 indirectLight = TEX_READ_2D(texAccum, gid).rgb;
    float3 albedo = TEX_READ_2D(texAlbedo, gid).rgb;
    float depth = TEX_READ_2D(texDepth, gid).r;
    
    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 screenSize = float2(width, height);
    float3 fog = SampleFogBilateral(texVolumetric, texDepth, uv, depth, screenSize);

    // Apply material (Linear Space)
    float3 totalIrradiance = directLight + indirectLight;
    float3 linearColor = totalIrradiance * albedo + fog;

    // Fog Logic
    if (depth < 50000.0f) {
        const float fogStart = COMPOSITE_FOG_START;
        const float fogDensity = COMPOSITE_FOG_DENSITY;
        float dist = max(depth - fogStart, 0.0f);
        float fogFactor = 1.0f - exp(-dist * fogDensity);
        float3 fogColor = float3(COMPOSITE_FOG_COLOR);
        linearColor = mix(linearColor, fogColor, fogFactor);
    }

    // Auto-exposure
    float avgLum = exposure.sceneLuminance;
    float exposureScale = 0.15f / (max(avgLum, 0.001f));
    linearColor *= exposureScale;

    // Saturation Boost
    linearColor = applySaturation(linearColor, (depth > 50000.0f) ? SKY_IMAGE_SATURATION : IMAGE_SATURATION);

    // Tone Mapping
    float3 toneMapped = ACESFilm(linearColor);

    // Gamma Correction
    float3 finalColor = LinearToSRGB(toneMapped);

    TEX_WRITE_2D(texFinal, float4(finalColor, 1.0f), gid);
}
