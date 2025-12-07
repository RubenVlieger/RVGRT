#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
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
        const float fogStart = 60.0f;
        const float fogDensity = 0.0002f; 
        float dist = max(depth - fogStart, 0.0f);
        float fogFactor = 1.0f - exp(-dist * fogDensity);
        float3 fogColor = float3(0.5f, 0.7f, 0.9f); 
        linearColor = mix(linearColor, fogColor, fogFactor);
    }

    // 5. COLOR GRADING (UPDATED with Auto-Exposure)
    
    // Retrieve calculated luminance from previous pass
    float avgLum = exposure.sceneLuminance;
    
    // Standard Exposure Formula:
    // Middle Grey (0.18 or 0.5 depending on calibration) / Average Luminance
    // We adjust the key value (1.0f) to taste.
    float exposureScale = 1.4f / (max(avgLum, 0.001f));
    
    linearColor *= exposureScale;

    // Saturation Boost 
    linearColor = applySaturation(linearColor, (depth > 50000.0f) ? 1.05f : 1.4f); 

    // Tone Mapping (ACES)
    float3 toneMapped = ACESFilm(linearColor);

    // Gamma Correction (Linear -> sRGB)
    float3 finalColor = LinearToSRGB(toneMapped);

    texFinal.write(float4(finalColor, 1.0f), gid);
}

kernel void distApproximationKernel( //1,5 gigaray per second
    texture2d<float, access::write> distTex [[texture(0)]],
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame [[buffer(1)]],
    
    texture3d<uint, access::read> bitsTex [[texture(2)]],
    texture3d<float, access::sample> csdf [[texture(3)]], 
    uint2 gid [[thread_position_in_grid]])
{
    uint width = distTex.get_width();
    uint height = distTex.get_height();
    
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f; 
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Move to start distance
    float3 currentPos = camera.position;

    // Precompute DDA constants (invariant for the ray)
    const float3 deltaDist = make_float3(
        abs(dir.x) > 1e-5f ? abs(1.0f / dir.x) : 1.0e30f,
        abs(dir.y) > 1e-5f ? abs(1.0f / dir.y) : 1.0e30f,
        abs(dir.z) > 1e-5f ? abs(1.0f / dir.z) : 1.0e30f
    );

    const int3 step = make_int3(
        dir.x > 0.0f ? 1 : -1,
        dir.y > 0.0f ? 1 : -1,
        dir.z > 0.0f ? 1 : -1
    );

    for (int majorIteration = 0; majorIteration < 10; majorIteration++)
    {
        currentPos = approximateCSDF(currentPos, dir, csdf);

        float3 fpos = floor3(currentPos);
        int3 ipos = to_int3(currentPos);
        
        float3 tMax;
        tMax.x = ((step.x > 0) ? (fpos.x + 1.0f - currentPos.x) : (currentPos.x - fpos.x)) * deltaDist.x;
        tMax.y = ((step.y > 0) ? (fpos.y + 1.0f - currentPos.y) : (currentPos.y - fpos.y)) * deltaDist.y;
        tMax.z = ((step.z > 0) ? (fpos.z + 1.0f - currentPos.z) : (currentPos.z - fpos.z)) * deltaDist.z;

        int mask = -1;
        float distTraveledInDDA = 0.0f;
        bool hitFound = false;

        for (int i = 0; i < 8; i++) 
        {            
            if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 || 
                ipos.x >= (int)SIZEX || ipos.y >= (int)SIZEY || ipos.z >= (int)SIZEZ) {
                majorIteration = 10;
                i = 10;
                break;
            }

            if (IsSolid(ipos, bitsTex)) 
            {
                float tVal = 0.0f;
                if (mask == 0) {
                    tVal = tMax.x - deltaDist.x;
                } else if (mask == 1) {
                    tVal = tMax.y - deltaDist.y;
                } else {
                    tVal = tMax.z - deltaDist.z;
                }
                distTex.write(float4(length(currentPos + dir * tVal - camera.position), 0, 0, 0), gid);
                return;
            }
            
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) { 
                    distTraveledInDDA = tMax.x;
                    tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; 
                } else { 
                    distTraveledInDDA = tMax.z;
                    tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
                }
            } else {
                if (tMax.y < tMax.z) { 
                    distTraveledInDDA = tMax.y;
                    tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; 
                } else { 
                    distTraveledInDDA = tMax.z;
                    tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
                }
            }
        }
        currentPos += dir * (distTraveledInDDA + 0.0001f);
    }        
    distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
}