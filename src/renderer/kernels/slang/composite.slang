// Composite
// Final composition pass. Combines direct light, indirect light, albedo,
// and volumetric fog. Applies tone mapping and gamma correction.

#include "ShaderTypes.slang"
#include "Settings.slang"

RWTexture2D<float4> texFinal;
Texture2D<float4> texDirect;
Texture2D<float4> texAccum;
Texture2D<float4> texAlbedo;
Texture2D<float4> texDepth;
Texture2D<float4> texVolumetric;
SamplerState fogSampler;

RWStructuredBuffer<ExposureData> exposure;

// Standard ACES fitted tone mapper
static float3 ACESFilm(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

static float3 LinearToSRGB(float3 color) {
    return select(color <= 0.0031308f, 12.92f * color, 1.055f * pow(color, 1.0f / 2.4f) - 0.055f);
}

static float3 applySaturation(float3 color, float saturation) {
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    return lerp(float3(luma), color, saturation);
}

static float3 SampleFogBilateral(
    Texture2D<float4> texVolumetric,
    Texture2D<float4> texDepth,
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
            int2 coord = basePos + int2(x, y);
            
            coord.x = max(0, min(coord.x, int(lowResSize.x) - 1));
            coord.y = max(0, min(coord.y, int(lowResSize.y) - 1));
            
            float3 fogColor = texVolumetric[coord].rgb;
            uint2 depthCoord = uint2(coord) * 2;
            float neighborDepth = texDepth[depthCoord].r;
            
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

[shader("compute")]
[numthreads(16, 16, 1)]
void Composite(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texFinal.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    // Gather Data
    float3 directLight = texDirect[gid.xy].rgb;
    float3 indirectLight = texAccum[gid.xy].rgb;
    float3 albedo = texAlbedo[gid.xy].rgb;
    float depth = texDepth[gid.xy].r;
    
    float2 uv = (float2(gid.xy) + 0.5f) / float2(dim);
    float2 screenSize = float2(dim);
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
        linearColor = lerp(linearColor, fogColor, fogFactor);
    }

    // Auto-exposure
    float avgLum = exposure[0].sceneLuminance;
    float exposureScale = 0.15f / (max(avgLum, 0.001f));
    linearColor *= exposureScale;

    // Saturation Boost
    linearColor = applySaturation(linearColor, (depth > 50000.0f) ? SKY_IMAGE_SATURATION : IMAGE_SATURATION);

    // Tone Mapping
    float3 toneMapped = ACESFilm(linearColor);

    // Gamma Correction
    float3 finalColor = LinearToSRGB(toneMapped);

    texFinal[gid.xy] = float4(finalColor, 1.0f);
}

// Fallback Blit (Upscaling + Format Conversion)
// Used when MetalFX is disabled to convert RGBA16Float to RGBA8Unorm
// and upscale from render resolution to output resolution.

Texture2D<float4> texSrc;
SamplerState blitSampler;
RWTexture2D<float4> texDst;

[shader("compute")]
[numthreads(16, 16, 1)]
void FallbackBlit(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texDst.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    float2 uv = (float2(gid.xy) + 0.5f) / float2(dim);
    float4 color = texSrc.Sample(blitSampler, uv);
    texDst[gid.xy] = color;
}
