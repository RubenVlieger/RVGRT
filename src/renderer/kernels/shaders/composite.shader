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

KERNEL(Composite)(
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texFinal, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDirect, 1),
    PARAM_TEXTURE_READ(tex2d_f32_r, texAccum, 2),
    PARAM_TEXTURE_READ(tex2d_f32_r, texAlbedo, 3),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 4),
    PARAM_TEXTURE_READ(tex2d_f32_s, texVolumetric, 5),
    
    PARAM_BUFFER(ExposureData, exposure, 0),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texFinal);
    int width = texFinal.get_width();
    int height = texFinal.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Gather Data
#if defined(PLATFORM_METAL)
    float3 directLight = TEX_READ_2D(texDirect, gid).rgb;
    float3 indirectLight = TEX_READ_2D(texAccum, gid).rgb;
    float3 albedo = TEX_READ_2D(texAlbedo, gid).rgb;
    float depth = TEX_READ_2D(texDepth, gid).r;
#else
    float4 direct4 = TEX_READ_2D(texDirect, gid);
    float4 indirect4 = TEX_READ_2D(texAccum, gid);
    float4 albedo4 = TEX_READ_2D(texAlbedo, gid);
    float4 depth4 = TEX_READ_2D(texDepth, gid);
    float3 directLight = make_float3(direct4.x, direct4.y, direct4.z);
    float3 indirectLight = make_float3(indirect4.x, indirect4.y, indirect4.z);
    float3 albedo = make_float3(albedo4.x, albedo4.y, albedo4.z);
    float depth = depth4.x;
#endif
    
#if defined(PLATFORM_METAL)
    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 screenSize = float2(width, height);
#else
    float2 uv = make_float2((gid.x + 0.5f) / (float)width, (gid.y + 0.5f) / (float)height);
    float2 screenSize = make_float2((float)width, (float)height);
#endif

    // ===== SampleFogBilateral inlined =====
#if defined(PLATFORM_METAL)
    float2 lowResSize = screenSize * 0.5f;
    float2 pixelPos = uv * lowResSize - 0.5f;
    int2 basePos = int2(floor(pixelPos));
    float2 weights = fract(pixelPos);
#else
    float2 lowResSize = screenSize * 0.5f;
    float2 pixelPos = uv * lowResSize - make_float2(0.5f, 0.5f);
    int2 basePos = make_int2(floorf(pixelPos.x), floorf(pixelPos.y));
    float2 weights = pixelPos - floor(pixelPos);
#endif
    
#if defined(PLATFORM_METAL)
    float3 sumColor = float3(0.0f);
#else
    float3 sumColor = make_float3(0.0f, 0.0f, 0.0f);
#endif
    float sumWeight = 0.0f;
    
    for(int y = 0; y <= 1; y++) {
        for(int x = 0; x <= 1; x++) {
#if defined(PLATFORM_METAL)
            int2 offset = int2(x, y);
            int2 coord = basePos + offset;
#else
            int2 offset = make_int2(x, y);
            int2 coord = make_int2(basePos.x + offset.x, basePos.y + offset.y);
#endif
            
            coord.x = max(0, min(coord.x, (int)lowResSize.x - 1));
            coord.y = max(0, min(coord.y, (int)lowResSize.y - 1));
            
#if defined(PLATFORM_METAL)
            float3 fogColor = texVolumetric.read(uint2(coord)).rgb;
            uint2 depthCoord = uint2(coord) * 2;
            float neighborDepth = texDepth.read(depthCoord).r;
#else
            float4 fogColor4 = TEX_READ_2D(texVolumetric, coord);
            float3 fogColor = make_float3(fogColor4.x, fogColor4.y, fogColor4.z);
            
            uint2 depthCoord = make_uint2(coord.x * 2, coord.y * 2);
            float4 neighborDepth4 = TEX_READ_2D(texDepth, depthCoord);
            float neighborDepth = neighborDepth4.x;
#endif
            
            float depthDiff = fabsf(depth - neighborDepth);
            float depthWeight = 1.0f / (1.0f + depthDiff * 2.0f);
            
            float bilinearWeight = (x == 0 ? (1.0f - weights.x) : weights.x) * 
                                   (y == 0 ? (1.0f - weights.y) : weights.y);
            
            float combinedWeight = depthWeight * bilinearWeight;
            
            sumColor = sumColor + fogColor * combinedWeight;
            sumWeight = sumWeight + combinedWeight;
        }
    }
    float3 fog = sumColor / (sumWeight + 0.0001f);

    // Apply material (Linear Space)
    float3 totalIrradiance = directLight + indirectLight;
    float3 linearColor = totalIrradiance * albedo + fog;

    // Fog Logic
    if (depth < 50000.0f) {
        const float fogStart = COMPOSITE_FOG_START;
        const float fogDensity = COMPOSITE_FOG_DENSITY;
#if defined(PLATFORM_METAL)
        float dist = max(depth - fogStart, 0.0f);
#else
        float dist = fmaxf(depth - fogStart, 0.0f);
#endif
        float fogFactor = 1.0f - expf(-dist * fogDensity);
#if defined(PLATFORM_METAL)
        float3 fogColor = float3(COMPOSITE_FOG_COLOR);
#else
        float3 fogColor = make_float3(0.5f, 0.7f, 0.9f);
#endif
        linearColor = mix(linearColor, fogColor, fogFactor);
    }

    // Auto-exposure
    float avgLum = exposure->sceneLuminance;
    float exposureScale = 0.15f / (fmaxf(avgLum, 0.001f));
#if defined(PLATFORM_METAL)
    linearColor *= exposureScale;
#else
    linearColor = linearColor * exposureScale;
#endif

    // Saturation Boost
#if defined(PLATFORM_METAL)
    float luma = dot(linearColor, float3(0.2126f, 0.7152f, 0.0722f));
    float3 saturated = mix(float3(luma), linearColor, (depth > 50000.0f) ? SKY_IMAGE_SATURATION : IMAGE_SATURATION);
#else
    // Manual dot product for CUDA device code
    float luma = linearColor.x * 0.2126f + linearColor.y * 0.7152f + linearColor.z * 0.0722f;
    float3 saturated = mix(make_float3(luma, luma, luma), linearColor, (depth > 50000.0f) ? SKY_IMAGE_SATURATION : IMAGE_SATURATION);
#endif

    // Tone Mapping (ACES)
#if defined(PLATFORM_METAL)
    float3 toneMapped = saturate((saturated*(2.51f*saturated+0.03f))/(saturated*(2.43f*saturated+0.59f)+0.14f));
#else
    float3 toneMapped = saturate((saturated*(2.51f*saturated+make_float3(0.03f)))/(saturated*(2.43f*saturated+make_float3(0.59f))+make_float3(0.14f)));
#endif

    // Gamma Correction (Linear to SRGB)
#if defined(PLATFORM_METAL)
    float3 finalColor = select(1.055f * pow(toneMapped, 1.0f / 2.4f) - 0.055f,
                                12.92f * toneMapped,
                                toneMapped <= 0.0031308f);
#else
    float3 finalColor;
    finalColor.x = (toneMapped.x <= 0.0031308f) ? (12.92f * toneMapped.x) : (1.055f * powf(toneMapped.x, 1.0f / 2.4f) - 0.055f);
    finalColor.y = (toneMapped.y <= 0.0031308f) ? (12.92f * toneMapped.y) : (1.055f * powf(toneMapped.y, 1.0f / 2.4f) - 0.055f);
    finalColor.z = (toneMapped.z <= 0.0031308f) ? (12.92f * toneMapped.z) : (1.055f * powf(toneMapped.z, 1.0f / 2.4f) - 0.055f);
#endif

#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texFinal, float4(finalColor, 1.0f), gid);
#else
    // RGBA8 format for final output (LDR)
    uchar4 finalUChar4;
    finalUChar4.x = (unsigned char)(fminf(fmaxf(finalColor.x * 255.0f, 0.0f), 255.0f));
    finalUChar4.y = (unsigned char)(fminf(fmaxf(finalColor.y * 255.0f, 0.0f), 255.0f));
    finalUChar4.z = (unsigned char)(fminf(fmaxf(finalColor.z * 255.0f, 0.0f), 255.0f));
    finalUChar4.w = 255;
    TEX_WRITE_2D_RGBA8(texFinal, finalUChar4, gid);
#endif
}
