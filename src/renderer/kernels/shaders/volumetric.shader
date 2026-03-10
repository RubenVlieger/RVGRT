#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "raytracing_functions.h"
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
#include "renderer/shader_settings.h"

// ============================================================================
// KERNEL: Volumetric Fog
// 
// Ray-marches through the scene to accumulate volumetric lighting.
// Uses temporal reprojection for stability.
// ============================================================================

inline float InterleavedGradientNoise(float2 pos) {
    return MATH_FRACT(52.9829189f * MATH_FRACT(0.06711056f * pos.x + 0.00583715f * pos.y));
}

inline float phaseFunction(float3 viewDir, float3 lightDir, float g) {
    float cosTheta = dot(viewDir, lightDir);
    float denom = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g) / (4.0f * 3.14159f * pow(denom, 1.5f));
}

KERNEL(VolumetricFog)(
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texVolumetric, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 1),
    PARAM_TEXTURE_READ(tex2d_f32_s, texHistory, 2),
    
    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),
    
    PARAM_TEXTURE_READ(tex3d_u32, indirection, 3),
    PARAM_BUFFER(SectorInfo, sectorBuffer, 3),
    PARAM_BUFFER(ulong, occupancyBuffer, 4),
    PARAM_BUFFER(ulong, sectorMaskBuffer, 6),
    PARAM_CONSTANT(CharacterGPUData, charData, 7),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texVolumetric);
    int width = texVolumetric.get_width();
    int height = texVolumetric.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

#if !VOLUMETRIC_FOG
    TEX_WRITE_2D(texVolumetric, float4(0.0f), gid);
    return;
#endif

    // Half-resolution coordinate
#if defined(PLATFORM_METAL)
    uint2 fullResCoord = gid * 2;
#else
    uint2 fullResCoord = AS_UINT2(gid) * make_uint2(2, 2);
#endif

    // Clamp to valid depth coord
    if (fullResCoord.x >= width * 2) fullResCoord.x = width * 2 - 1;
    if (fullResCoord.y >= height * 2) fullResCoord.y = height * 2 - 1;

    float depth = TEX_READ_2D(texDepth, fullResCoord).r;
    float2 uv = (AS_FLOAT2(gid) + 0.5f) / make_float2(width, height);

    float fogMaxDist = VOLUMETRIC_MAXDIST;
    float clampedDepth = min(depth, fogMaxDist);
    
    float3 endPos = reconstructPos(clampedDepth, uv, camera);
    float3 startPos = camera.position;
    
    float3 rayVec = endPos - startPos;
    float rayLength = length(rayVec);
    float3 rayDir = normalize(rayVec);

    // Ray marching with dithering
    float dither = InterleavedGradientNoise(AS_FLOAT2(gid) + make_float2(frame.time * 5.588f));
    
    const int STEPS = VOLUMETRIC_STEPS;
    float stepSize = rayLength / float(STEPS);
    float currentT = stepSize * dither;
    
    float3 accumulatedLight = float3(0.0f);
    float accumulatedTransmittance = 1.0f;
    
    const float density = FOG_DENSITY;
    const float3 fogColor = float3(FOG_COLOR);
    const float anisotropy = FOG_ANISOTROPY;
    float phase = phaseFunction(rayDir, frame.sunDirection, anisotropy);
    float3 sunColor = float3(c_sunColor);

    for(int i = 0; i < STEPS; i++) {
        float3 pos = startPos + rayDir * currentT;
        
        bool isShadowed = false;
        if (currentT > 2.0f && currentT < 200.0f && phase > 0.04f) {
#if SHADOWS
            isShadowed = traceShadow(pos, frame.sunDirection, VOLUMETRIC_SHADOW_MAXDIST,
                                    VOLUMETRIC_SHADOW_STEPS, indirection, sectorBuffer,
                                    occupancyBuffer, 0, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
#else
            isShadowed = false;
#endif
        }

        if (!isShadowed) {
            accumulatedLight += sunColor * phase * density * accumulatedTransmittance * stepSize;
        }
        accumulatedLight += (fogColor * 0.05f) * density * accumulatedTransmittance * stepSize;
        accumulatedTransmittance *= exp(-density * stepSize);
        currentT += stepSize;
    }

    // Reprojection
    float4 prevClip = camera.prevUnjitteredViewProjection * float4(endPos, 1.0f);
    
    float3 history = float3(0.0f);
    float blendFactor = 0.0f;

    if (prevClip.w > 0.01f) {
        float2 prevNDC = prevClip.xy / prevClip.w;
        
        float2 prevUV;
        prevUV.x = prevNDC.x * 0.5f + 0.5f;
        prevUV.y = 0.5f - prevNDC.y * 0.5f;

        if (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f) {
            DECLARE_SAMPLER(sLinear, linear, clamp_to_edge);
            history = TEX_SAMPLE_2D(texHistory, prevUV).rgb;
            
            float diff = length(history - accumulatedLight);
            blendFactor = 0.8f;
            if (diff > 1.0f) blendFactor = 0.4f;
        }
    }

    float3 result = mix(accumulatedLight, history, blendFactor);
    TEX_WRITE_2D(texVolumetric, float4(result, 1.0f), gid);
}
